#rdi datare ptr
#rsi dataim ptr
#rdx om ptr
.globl reim_fft4_avx_fma
reim_fft4_avx_fma:
vmovupd     (%rdi),%xmm0       # ra0
vmovupd     0x10(%rdi),%xmm1   # ra2
vmovupd     (%rsi),%xmm4       # ia0
vmovupd     0x10(%rsi),%xmm5   # ia2

1:
vmovupd     0x0(%rdx),%xmm12
vshufpd     $3, %xmm12, %xmm12, %xmm13  # ymm13: omai
vshufpd     $0, %xmm12, %xmm12, %xmm12  # ymm12: omar
vmulpd      %xmm5,%xmm13,%xmm8 # ia0.omai (tw)
vmulpd      %xmm1,%xmm13,%xmm10 # ra0.omai (tw)
vfmsub231pd %xmm1,%xmm12,%xmm8 # rprod0 (tw)
vfmadd231pd %xmm5,%xmm12,%xmm10 # iprod0 (tw)
vsubpd      %xmm8,%xmm0,%xmm1
vsubpd      %xmm10,%xmm4,%xmm5
vaddpd      %xmm8,%xmm0,%xmm0
vaddpd      %xmm10,%xmm4,%xmm4

2:
vmovupd     0x10(%rdx),%xmm12             # om: r,i
vshufpd     $1, %xmm12, %xmm12, %xmm13    # omi: i,r
xorpd       fft4neg(%rip),%xmm12          # omr: r,-i

vunpckhpd %xmm1,%xmm0,%xmm8
vunpckhpd %xmm5,%xmm4,%xmm10
vunpcklpd %xmm1,%xmm0,%xmm0
vunpcklpd %xmm5,%xmm4,%xmm2

vmulpd      %xmm10,%xmm13,%xmm4 # ia0.omi (tw)
vmulpd      %xmm8,%xmm13,%xmm6 # ra0.omi (tw)
vfmsub231pd %xmm8,%xmm12,%xmm4 # rprod0 (tw)
vfmadd231pd %xmm10,%xmm12,%xmm6 # iprod0 (tw)
vsubpd      %xmm4,%xmm0,%xmm8
vsubpd      %xmm6,%xmm2,%xmm10
vaddpd      %xmm4,%xmm0,%xmm0
vaddpd      %xmm6,%xmm2,%xmm2

vunpckhpd %xmm8,%xmm0,%xmm1   # (0,4) -> (0,1)
vunpckhpd %xmm10,%xmm2,%xmm5    # (2,6) -> (2,3)
vunpcklpd %xmm8,%xmm0,%xmm0
vunpcklpd %xmm10,%xmm2,%xmm4

4:
vmovupd     %xmm0,(%rdi)       # ra0
vmovupd     %xmm1,0x10(%rdi)   # ra4
vmovupd     %xmm4,(%rsi)       # ia0
vmovupd     %xmm5,0x10(%rsi)   # ia4
ret

/* Constants for YMM */
.balign 32
fft4neg: .quad 0x0, 0x8000000000000000

.size	reim_fft4_avx_fma, .-reim_fft4_avx_fma
.section .note.GNU-stack,"",@progbits
