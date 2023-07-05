#ifndef SPQLIOS_COMMONS_H
#define SPQLIOS_COMMONS_H

#ifdef __cplusplus
#include <cstdint>
#define EXPORT extern "C"
#define EXPORT_DECL extern "C"
#else
#include <stdint.h>
#define EXPORT
#define EXPORT_DECL extern
#endif

#endif  // SPQLIOS_COMMONS_H
