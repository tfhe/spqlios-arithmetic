#rdi datare ptr
#rsi dataim ptr
#rdx om ptr
.globl reim_ifft4_avx_fma
reim_ifft4_avx_fma:
vmovupd     (%rdi),%xmm0       # ra0
vmovupd     0x10(%rdi),%xmm1   # ra2
vmovupd     (%rsi),%xmm4       # ia0
vmovupd     0x10(%rsi),%xmm5   # ia2

1:
vmovupd     0x00(%rdx),%xmm12             # omr: r,i
vshufpd     $1, %xmm12, %xmm12, %xmm13
xorpd       ifft4neg(%rip),%xmm13         # omi: i,-r

vunpckhpd %xmm1,%xmm0,%xmm8
vunpckhpd %xmm5,%xmm4,%xmm10
vunpcklpd %xmm1,%xmm0,%xmm0
vunpcklpd %xmm5,%xmm4,%xmm2

#twiddle  R:(0,8) I:(2,10) with omega=(12,13)
vsubpd      %xmm8,%xmm0,%xmm4  # rtw
vsubpd      %xmm10,%xmm2,%xmm6 # itw
vaddpd      %xmm8,%xmm0,%xmm0
vaddpd      %xmm10,%xmm2,%xmm2
#mul (4,6) with (12,13) res in (8,10)
vmulpd      %xmm6,%xmm13,%xmm8  # itw.omi (tw)
vmulpd      %xmm4,%xmm13,%xmm10 # rtw.omi (tw)
vfmsub231pd %xmm4,%xmm12,%xmm8  # rprod0 (tw)
vfmadd231pd %xmm6,%xmm12,%xmm10 # iprod0 (tw)

vunpckhpd %xmm8,%xmm0,%xmm1   # (0,4) -> (0,1)
vunpckhpd %xmm10,%xmm2,%xmm5    # (2,6) -> (2,3)
vunpcklpd %xmm8,%xmm0,%xmm0
vunpcklpd %xmm10,%xmm2,%xmm4

2:
vmovupd     0x10(%rdx),%xmm12
vshufpd     $3, %xmm12, %xmm12, %xmm13  # ymm13: omai
vshufpd     $0, %xmm12, %xmm12, %xmm12  # ymm12: omar

#twiddle  R:(0,1) I:(4,5) with omega=(12,13)
vsubpd      %xmm1,%xmm0,%xmm8  # rtw
vsubpd      %xmm5,%xmm4,%xmm9 # itw
vaddpd      %xmm1,%xmm0,%xmm0
vaddpd      %xmm5,%xmm4,%xmm4
#mul (8,9) with (12,13) res in (1,5)
vmulpd      %xmm9,%xmm13,%xmm1  # itw.omi (tw)
vmulpd      %xmm8,%xmm13,%xmm5  # rtw.omi (tw)
vfmsub231pd %xmm8,%xmm12,%xmm1  # rprod0 (tw)
vfmadd231pd %xmm9,%xmm12,%xmm5  # iprod0 (tw)

4:
vmovupd     %xmm0,(%rdi)       # ra0
vmovupd     %xmm1,0x10(%rdi)   # ra4
vmovupd     %xmm4,(%rsi)       # ia0
vmovupd     %xmm5,0x10(%rsi)   # ia4
ret

/* Constants for YMM */
.balign 32
ifft4neg: .quad 0x0, 0x8000000000000000

.size	reim_ifft4_avx_fma, .-reim_ifft4_avx_fma
.section .note.GNU-stack,"",@progbits
