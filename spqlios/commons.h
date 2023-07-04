#ifndef SPQLIOS_COMMONS_H
#define SPQLIOS_COMMONS_H

#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define EXPORT extern "C"
#define EXPORT_DECL extern "C"
#else
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define EXPORT
#define EXPORT_DECL extern
#define nullptr 0x0;
#endif

#endif  // SPQLIOS_COMMONS_H
