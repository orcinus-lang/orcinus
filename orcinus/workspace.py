# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

import io
import logging
import os
import urllib.parse
import urllib.parse
import urllib.parse
import weakref
from typing import Optional, MutableMapping
from typing import Sequence

from orcinus.diagnostics import DiagnosticManager, Diagnostic
from orcinus.exceptions import OrcinusError
from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.semantic import SemanticModel, SemanticContext
from orcinus.signals import Signal
from orcinus.syntax import SyntaxTree, SyntaxContext
from orcinus.utils import cached_property

logger = logging.getLogger('orcinus.workspace')


class Workspace:
    """
    Active representation of collection of projects
    """
    packages: Sequence[Package]

    on_document_create: Signal  # (document: Document) -> void
    on_document_remove: Signal  # (document: Document) -> void
    on_document_analyze: Signal  # (document: Document) -> void

    def __init__(self, paths: Sequence[str] = None):
        paths = list(() or paths)

        # Standard library path
        stdlib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stdlib'))
        if stdlib_path not in paths:
            paths.insert(0, stdlib_path)

        self.packages = [
            Package(self, os.path.abspath(urllib.parse.urlparse(path).path)) for path in paths
        ]

        # signals
        self.on_document_create = Signal()
        self.on_document_remove = Signal()
        self.on_document_analyze = Signal()

    def get_package_for_document(self, doc_uri: str):
        url = urllib.parse.urlparse(doc_uri)
        fullname = os.path.abspath(url.path)
        for package in self.packages:
            if fullname.startswith(package.path):
                return package

        raise OrcinusError(f"Not found file `{url.path}` in packages")

    def get_or_create_document(self, doc_uri: str) -> Document:
        """
        Return a managed document if-present, else create one pointing at disk.
        """
        package = self.get_package_for_document(doc_uri)
        return package.get_or_create_document(doc_uri)

    def get_document(self, doc_uri: str) -> Optional[Document]:
        """ Returns a managed document if-present, otherwise None """
        package = self.get_package_for_document(doc_uri)
        return package.get_document(doc_uri)

    def create_document(self, doc_uri, source=None, version=None) -> Document:
        """ Create new document """
        package = self.get_package_for_document(doc_uri)
        return package.create_document(doc_uri, source, version)

    def update_document(self, doc_uri: str, source=None, version=None) -> Document:
        """ Update source of document """
        package = self.get_package_for_document(doc_uri)
        return package.update_document(doc_uri, source, version)

    def unload_document(self, doc_uri: str):
        """ Unload document from package """
        package = self.get_package_for_document(doc_uri)
        return package.unload_document(doc_uri)

    def load_document(self, module_name: str) -> Document:
        """
        Load document for module

        :param module_name: Module name
        :return: Document
        """
        for package in self.packages:
            doc_uri = convert_filename(module_name, package.path)
            try:
                return package.get_or_create_document(doc_uri)
            except IOError:
                logger.debug(f"Not found module `{module_name}` in file `{doc_uri}`")
                pass  # Continue

        raise OrcinusError(f'Not found module {module_name}')


class Package:
    """ Instance of this class is managed single package """

    def __init__(self, workspace: Workspace, path: str):
        self.__workspace = weakref.ref(workspace)
        self.path = path
        self.documents: MutableMapping[str, Document] = {}

    @property
    def workspace(self) -> Workspace:
        return self.__workspace()

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def get_module_name(self, filename):
        fullname = os.path.abspath(filename)
        if fullname.startswith(self.path):
            return convert_module_name(fullname, self.path)

        raise OrcinusError(f"Not found file `{filename}` in packages")

    # def get_filename(self, filename):

    def get_or_create_document(self, doc_uri: str) -> Document:
        """
        Return a managed document if-present, else create one pointing at disk.
        """
        return self.get_document(doc_uri) or self.create_document(doc_uri)

    def get_document(self, doc_uri: str) -> Optional[Document]:
        """ Returns a managed document if-present, otherwise None """
        return self.documents.get(doc_uri)

    def create_document(self, doc_uri, source=None, version=None) -> Document:
        """ Create new document """
        url = urllib.parse.urlparse(doc_uri)
        name = self.get_module_name(url.path)

        if source is None:
            with open(url.path, 'r', encoding='utf-8') as stream:
                source = stream.read()
        document = Document(self, doc_uri, name=name, source=source, version=version)
        self.documents[doc_uri] = document
        self.workspace.on_document_create(document=document)
        return document

    def update_document(self, doc_uri: str, source=None, version=None) -> Document:
        """ Update source of document """
        document = self.get_document(doc_uri) or self.create_document(doc_uri, source, version)
        document.source = source
        self.documents[doc_uri] = document
        return document

    def unload_document(self, doc_uri: str):
        """ Unload document from package """
        try:
            document = self.documents[doc_uri]
            del self.documents[doc_uri]
        except KeyError:
            pass
        else:
            self.workspace.on_document_remove(document=document)

    def __str__(self) -> str:
        return f'{self.name} [{self.path}]'

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


class Document:
    def __init__(self, package: Package, uri: str, name: str, source: str = None, version: int = None,
                 diagnostics: DiagnosticManager = None):
        self.__package = weakref.ref(package)
        self.__diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.__uri = uri
        self.__name = name
        self.__source = source
        self.__version = version
        self.__tree = None
        self.__model = None
        self.__module = None

    @property
    def package(self) -> Package:
        return self.__package()

    @property
    def workspace(self) -> Workspace:
        return self.package.workspace

    @property
    def name(self) -> str:
        """ Return module name """
        return self.__name

    @property
    def uri(self):
        """ Returns uri for source path """
        return self.__uri

    @cached_property
    def path(self) -> str:
        url = urllib.parse.urlparse(self.uri)
        filename = os.path.abspath(url.path)
        return os.path.relpath(filename, self.package.path)

    @property
    def source(self) -> str:
        """ Returns source of document """
        return self.__source

    @source.setter
    def source(self, value: str):
        """ Change source of document """
        self.__source = value
        self.invalidate()

    @property
    def diagnostics(self) -> DiagnosticManager:
        """ Returns diagnostics manager for this document """
        return self.__diagnostics

    @property
    def tree(self) -> SyntaxTree:
        """ Returns syntax tree """
        if not self.__tree:
            context = SyntaxContext(self.diagnostics)
            stream = io.StringIO(self.source)
            scanner = Scanner(self.uri, diagnostics=self.diagnostics)
            parser = Parser(scanner, context)
            self.__tree = parser.parse()
        return self.__tree

    @property
    def model(self) -> SemanticModel:
        """ Returns semantic model """
        if not self.__model:
            try:
                context = SemanticContext(self.workspace, diagnostics=self.diagnostics)
                self.__model = context.open(self)
            except Diagnostic as ex:
                self.diagnostics.add(ex.location, ex.severity, ex.message, ex.source)
            finally:
                self.workspace.on_document_analyze(document=self)
        return self.__model

    @property
    def module(self) -> Module:
        """ Return semantic module for this document """
        if not self.__module:
            self.__module = self.model.module if self.model else None
        return self.__module

    def invalidate(self):
        """ Invalidate document, e.g. detach syntax tree or semantic model from this document """
        self.diagnostics.clear()
        self.__module = None
        self.__model = None
        self.__tree = None

    def __str__(self) -> str:
        return f'{self.package.name}::{self.name} [{self.path}]'

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


def convert_module_name(filename, path):
    fullname = os.path.abspath(filename)
    if not fullname.startswith(path):
        raise OrcinusError(f"Not found file `{filename}` in package `{path}`")

    module_name = fullname[len(path):]
    module_name, _ = os.path.splitext(module_name)
    module_name = module_name.strip(os.path.sep)
    module_name = module_name.replace(os.path.sep, '.')
    return module_name


def convert_filename(module_name, path):
    filename = module_name.replace('.', os.path.sep) + '.orx'
    return os.path.join(path, filename)
