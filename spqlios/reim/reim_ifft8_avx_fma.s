#rdi datare ptr
#rsi dataim ptr
#rdx om ptr
.globl reim_ifft8_avx_fma
reim_ifft8_avx_fma:
vmovupd     (%rdi),%ymm0       # ra0
vmovupd     0x20(%rdi),%ymm1   # ra4
vmovupd     (%rsi),%ymm4       # ia0
vmovupd     0x20(%rsi),%ymm5   # ia4

1:
vmovupd    0x00(%rdx),%ymm12               # omr: r0,r1,i0,i1
vperm2f128 $0x01,%ymm12,%ymm12,%ymm13
vxorpd     fft8neg2(%rip),%ymm13,%ymm13    # omi: i0,i1,-r0,-r1

vunpckhpd %ymm1,%ymm0,%ymm8
vunpckhpd %ymm5,%ymm4,%ymm10
vunpcklpd %ymm1,%ymm0,%ymm0
vunpcklpd %ymm5,%ymm4,%ymm2

#twiddle R:(%ymm0,%ymm8) I:(%ymm2,%ymm10)
vsubpd      %ymm8,%ymm0,%ymm4  # rtw
vsubpd      %ymm10,%ymm2,%ymm6 # itw
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm10,%ymm2,%ymm2
# mul 4,6 with 12,13 result in 8,10
vmulpd      %ymm6,%ymm13,%ymm8  # itw.omi (tw)
vmulpd      %ymm4,%ymm13,%ymm10 # rtw.omi (tw)
vfmsub231pd %ymm4,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm6,%ymm12,%ymm10 # iprod0 (tw)

vunpckhpd %ymm8,%ymm0,%ymm1   # (0,4) -> (0,1)
vunpckhpd %ymm10,%ymm2,%ymm5    # (2,6) -> (2,3)
vunpcklpd %ymm8,%ymm0,%ymm0
vunpcklpd %ymm10,%ymm2,%ymm4

2:
vmovapd     ifft8neg(%rip),%xmm13
vmovupd     0x20(%rdx),%xmm12             # om: r,i
xorpd       %xmm12,%xmm13                 # om: -r,i
vinsertf128 $1, %xmm13, %ymm12, %ymm12    # om: r,i,-r,i
vshufpd     $3, %ymm12, %ymm12, %ymm13    # omi: i,i,-r,-r
vshufpd     $12,  %ymm12, %ymm12, %ymm12  # omr: r,r,i,i

vperm2f128 $0x31,%ymm1,%ymm0,%ymm8   # ymm8 contains re to mul (tw)
vperm2f128 $0x31,%ymm5,%ymm4,%ymm10   # ymm10 contains im to mul (tw)
vperm2f128 $0x20,%ymm1,%ymm0,%ymm0   # ymm0 contains re to add (tw)
vperm2f128 $0x20,%ymm5,%ymm4,%ymm2   # ymm2 contains im to add (tw)

#twiddle R:(%ymm0,%ymm8) I:(%ymm2,%ymm10)
vsubpd      %ymm8,%ymm0,%ymm4  # rtw
vsubpd      %ymm10,%ymm2,%ymm6 # itw
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm10,%ymm2,%ymm2
# mul 4,6 with 12,13 result in 8,10
vmulpd      %ymm6,%ymm13,%ymm8  # itw.omi (tw)
vmulpd      %ymm4,%ymm13,%ymm10 # rtw.omi (tw)
vfmsub231pd %ymm4,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm6,%ymm12,%ymm10 # iprod0 (tw)

vperm2f128 $0x31,%ymm10,%ymm2,%ymm5
vperm2f128 $0x20,%ymm10,%ymm2,%ymm4
vperm2f128 $0x31,%ymm8,%ymm0,%ymm1
vperm2f128 $0x20,%ymm8,%ymm0,%ymm0

3:
vmovupd     0x30(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar

#twiddle R:(%ymm0,%ymm1) I:(%ymm4,%ymm5)
vsubpd      %ymm1,%ymm0,%ymm8  # rtw
vsubpd      %ymm5,%ymm4,%ymm10 # itw
vaddpd      %ymm1,%ymm0,%ymm0
vaddpd      %ymm5,%ymm4,%ymm4
# mul 8,10 with 12,13 result in 1,5
vmulpd      %ymm10,%ymm13,%ymm1  # itw.omi (tw)
vmulpd      %ymm8,%ymm13,%ymm5 # rtw.omi (tw)
vfmsub231pd %ymm8,%ymm12,%ymm1 # rprod0 (tw)
vfmadd231pd %ymm10,%ymm12,%ymm5 # iprod0 (tw)

4:
vmovupd     %ymm0,(%rdi)       # ra0
vmovupd     %ymm1,0x20(%rdi)   # ra4
vmovupd     %ymm4,(%rsi)       # ia0
vmovupd     %ymm5,0x20(%rsi)   # ia4
vzeroupper
ret

/* Constants for YMM */
.balign 32
fft8neg2: .quad 0x0, 0x0, 0x8000000000000000, 0x8000000000000000
ifft8neg: .quad 0x8000000000000000, 0x0

.size	reim_ifft8_avx_fma, .-reim_ifft8_avx_fma
.section .note.GNU-stack,"",@progbits
