/* Zig 0.14.1's GNU/Linux shared link omits crtbeginS and crtendS. Associate
 * C++ callbacks with this object and run them before the loader unmaps it. */
__attribute__((visibility("hidden"))) void *__dso_handle = &__dso_handle;
extern void __cxa_finalize(void *handle);

__attribute__((destructor))
static void finalize_cpp_statics(void) {
    __cxa_finalize(__dso_handle);
}
