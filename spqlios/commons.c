#include "commons.h"

#include <stdio.h>
#include <stdlib.h>

EXPORT void* spqlios_error(const char* error) {
  fputs(error, stderr);
  abort();
  return nullptr;
}
EXPORT void* spqlios_keep_or_free(void* ptr, void* ptr2) {
  if (!ptr2) {
    free(ptr);
  }
  return ptr2;
}