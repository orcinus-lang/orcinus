Exit with crash
==========

At call `orx_exit` runtime is failed to successfully exit.

Error is incorrect use of complex libraries as `gc`, `coro` and `uv`.  
 
Strace of error:

```
mmap(NULL, 8392704, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7f2092cba000
mprotect(0x7f2092cbb000, 8388608, PROT_READ|PROT_WRITE) = 0
clone(child_stack=0x7f20934b9fb0, flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, parent_tidptr=0x7f20934ba9d0, tls=0x7f20934ba700, child_tidptr=0x7f20934ba9d0) = 6846
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
mmap(0x7f20944fd000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2095655000
futex(0x7f209461d608, FUTEX_WAIT_PRIVATE, 0, NULL) = 0
futex(0x7f209461d620, FUTEX_WAIT_PRIVATE, 2, NULL) = 0
futex(0x7f209461d620, FUTEX_WAKE_PRIVATE, 1) = 0
rt_sigprocmask(SIG_BLOCK, NULL, [], 8)  = 0
futex(0x7f209461da48, FUTEX_WAKE_PRIVATE, 2147483647) = 2
futex(0x7f209461d620, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x7f209461da30, FUTEX_WAIT_PRIVATE, 2, NULL) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x7f209461da30, FUTEX_WAIT_PRIVATE, 3, NULL) = 0
futex(0x7f209461da4c, FUTEX_WAKE_PRIVATE, 2147483647) = 1
futex(0x7f209461d620, FUTEX_WAKE_PRIVATE, 1) = 1
futex(0x7f209461da48, FUTEX_WAIT_PRIVATE, 0, NULL) = 0
futex(0x7f209461d620, FUTEX_WAKE_PRIVATE, 1) = 0
futex(0x7f209461da30, FUTEX_WAIT_PRIVATE, 2, NULL) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x7f209461da30, FUTEX_WAIT_PRIVATE, 3, NULL) = -1 EAGAIN (Resource temporarily unavailable)
futex(0x7f209461da4c, FUTEX_WAKE_PRIVATE, 2147483647) = 2
mmap(0x7f2095665000, 352256, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2092c64000
mmap(0x7f2092cba000, 471040, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2092bf1000
rt_sigprocmask(SIG_BLOCK, ~[RTMIN RT_1], [], 8) = 0
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2095654000
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
clock_getres(CLOCK_MONOTONIC_COARSE, {tv_sec=0, tv_nsec=4000000}) = 0
rt_sigprocmask(SIG_BLOCK, ~[RTMIN RT_1], [], 8) = 0
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2095653000
rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
munmap(0x7f20945bd000, 2835200)         = 0
+++ killed by SIGSEGV (core dumped) +++
```
