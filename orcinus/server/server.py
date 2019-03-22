import json
import logging
import socket
from typing import Optional

from jsonrpc import Dispatcher, JSONRPCResponseManager
from jsonrpc.jsonrpc2 import JSONRPC20Request

from orcinus.diagnostics import DiagnosticManager
from orcinus.server.constants import TextDocumentSyncKind, DOCUMENT_PUBLISH_DIAGNOSTICS
from orcinus.server.converters import from_lsp_position, to_lsp_diagnostic, to_lsp_completion_kind
# from orcinus.syntax import AliasAST, ImportModuleAST
from orcinus.symbols import Type
from orcinus.syntax import SyntaxToken, ParameterNode
from orcinus.workspace import Workspace, Document

logger = logging.getLogger('orcinus')


class LanguageTCPServer:
    def __init__(self):
        # Create a TCP/IP socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def listen(self, hostname='0.0.0.0', port=10000):
        # Bind the socket to the port
        server_address = (hostname, port)
        logger.info('Starting Orcinus LSP server on {} port {}'.format(*server_address))
        self.server.bind(server_address)

        # Listen for incoming connections
        self.server.listen(1)

        try:
            while True:
                # Wait for a connection
                logger.debug('Waiting for a connection')
                try:
                    connection, client_address = self.server.accept()
                    self.process(connection, client_address)
                except KeyboardInterrupt:
                    return
        finally:
            self.server.close()

    def process(self, connection, client_address):
        try:
            logger.debug('Connection from {}:{}'.format(*client_address))
            client = LanguageTCPClient(self, connection)
            client.handle()
        except EOFError:
            logger.debug('Connection {}:{} closed'.format(*client_address))
        except Exception:
            logger.error("Language server shutting down for uncaught exception")
            raise
        finally:
            # Clean up the connection
            connection.close()


# noinspection PyPep8Naming
class LanguageTCPClient:
    def __init__(self, server, connection) -> None:
        super().__init__()

        self.__server = server  # type: LanguageTCPServer
        self.__connection = connection
        self.__reader = connection.makefile('r')
        self.__writer = connection.makefile('w')
        self.__workspace = None

        dispatcher = Dispatcher()
        self.dispatcher = dispatcher

        dispatcher.add_method(self.initialize)
        dispatcher.add_method(self.initialized)
        dispatcher.add_method(self.workspace_change_config, 'workspace/didChangeConfiguration')
        dispatcher.add_method(self.text_document_open, 'textDocument/didOpen')
        dispatcher.add_method(self.text_document_change, 'textDocument/didChange')
        dispatcher.add_method(self.text_document_close, 'textDocument/didClose')
        dispatcher.add_method(self.text_document_completion, 'textDocument/completion')

    @property
    def capabilities(self):
        return {
            'capabilities': {
                'textDocumentSync': TextDocumentSyncKind.Full,
                'completionProvider': {
                    'resolveProvider': False,
                    'triggerCharacters': ['.', ' ']
                },
                # 'definitionProvider': True,
                # 'documentSymbolProvider': True,
                # 'workspaceSymbolProvider': True,
                'workspace': {
                    'workspaceFolders': {
                        'supported': True,
                        # 'changeNotifications': True
                    }
                }
            }
        }

    @property
    def workspace(self) -> Optional[Workspace]:
        return self.__workspace

    def handle(self):
        while True:
            data = self.__read_message()
            response = JSONRPCResponseManager.handle(data, self.dispatcher)
            if response is not None:
                self.__write_message(data=response.data)

    def notify(self, method, params=None):
        """ Send a notification to the client, expects no response. """
        logger.debug("Sending notification %s", method)
        request = JSONRPC20Request(method=method, params=params, is_notification=True)
        self.__write_message(data=request.data)

    @staticmethod
    def __get_content_length(line):
        if line.startswith('Content-Length: '):
            _, value = line.split('Content-Length: ')
            value = value.strip()
            try:
                return int(value)
            except ValueError:
                raise ValueError("Invalid Content-Length header: {}".format(value))

    def __read_line(self):
        pass

    def __read_message(self) -> bytes:
        line = self.__reader.readline()
        if not line:
            raise EOFError()

        content_length = self.__get_content_length(line)

        # Blindly consume all header lines
        while line and line.strip():
            line = self.__reader.readline()

        if not line:
            raise EOFError()

        # Grab the body
        return self.__reader.read(content_length)

    def __write_message(self, *, data: dict = None, body: str = None):
        if data:
            body = json.dumps(data, separators=(",", ":"))
        elif not body:
            return
        # log.debug("Send message: %s", data)
        content_length = len(body)
        response = (
            "Content-Length: {}\r\n"
            "Content-Type: application/vscode-jsonrpc; charset=utf8\r\n\r\n"
            "{}".format(content_length, body)
        )
        self.__writer.write(response)
        self.__writer.flush()

    def initialize(self, processId, rootPath, rootUri, initializationOptions=None, workspaceFolders=None, **kwargs):
        logger.info("Receive IDE initialize parameters")
        if processId:
            logger.debug(f"Start from process {processId}")

        if workspaceFolders:
            self.__workspace = Workspace(workspaceFolders)
        else:
            self.__workspace = Workspace([rootUri])

        # InitializeResult
        return self.capabilities

    def initialized(self):
        logger.info("Server initialized for IDE")
        pass

    def workspace_change_config(self, *args, **kwargs):
        logger.info("Receive workspace configuration changes")
        pass

    def text_document_open(self, textDocument):
        logger.info(f"Open document: {textDocument['uri']}")
        document = self.workspace.update_document(textDocument['uri'], textDocument['text'], textDocument['version'])
        self.analyze(document)

    def text_document_change(self, textDocument, contentChanges):
        logger.debug(f"Change document: {textDocument['uri']}")
        document = self.workspace.get_or_create_document(textDocument['uri'])
        document.source = contentChanges[0]['text']
        self.analyze(document)

    def text_document_close(self, textDocument):
        logger.info(f"Close document: {textDocument['uri']}")
        self.workspace.unload_document(textDocument['uri'])

    def text_document_completion(self, textDocument, position, context=None):
        document = self.workspace.get_or_create_document(textDocument['uri'])
        position = from_lsp_position(position, is_end=True)
        logger.debug(f"Completion document: {textDocument['uri']} on position {position}")

        node = None
        founded_symbol = document.syntax_tree.find_contains(position)
        if isinstance(founded_symbol, SyntaxToken):
            parent = founded_symbol.parent
            if isinstance(parent, ParameterNode):
                node = parent
            else:
                node = parent.find_closest(lambda n: isinstance(node, ParameterNode))

        items = []
        # if isinstance(node, ParameterNode):
        #     if node.token_colon is founded_symbol:
        #         # a:<-
        #     elif node.type
        types = document.semantic_model.get_types()
        types = sorted(set(types), key=lambda t: t.name)

        for type in types:  # type: Type
            assert isinstance(type.name, str)
            items.append({
                'label': type.name,
                'kind': to_lsp_completion_kind(type)
            })

        # if isinstance(node, (AliasAST, ImportModuleAST)):
        #     items.append({
        #         'label': 'system',
        #         'kind': CompletionItemKind.Module
        #     })
        #     items.append({
        #         'label': 'io',
        #         'kind': CompletionItemKind.Module
        #     })
        #     items.append({
        #         'label': '_io',
        #         'kind': CompletionItemKind.Module,
        #         'deprecated': True,
        #     })

        return {
            'isIncomplete': False,
            'items': items
        }

    def publish_diagnostics(self, doc_uri: str, diagnostics: DiagnosticManager):
        diagnostics = tuple(map(to_lsp_diagnostic, diagnostics))
        self.notify(DOCUMENT_PUBLISH_DIAGNOSTICS, params={'uri': doc_uri, 'diagnostics': diagnostics})

    def analyze(self, document: Document):
        document.analyze()
        self.publish_diagnostics(document.uri, document.diagnostics)
