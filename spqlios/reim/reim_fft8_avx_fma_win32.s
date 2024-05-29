#rdi datare ptr
#rsi dataim ptr
#rdx om ptr
.globl reim_fft8_avx_fma
reim_fft8_avx_fma:
vmovupd     (%rdi),%ymm0       # ra0
vmovupd     0x20(%rdi),%ymm1   # ra4
vmovupd     (%rsi),%ymm4       # ia0
vmovupd     0x20(%rsi),%ymm5   # ia4

1:
vmovupd     0x0(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar
vmulpd      %ymm5,%ymm13,%ymm8 # ia0.omai (tw)
vmulpd      %ymm1,%ymm13,%ymm10 # ra0.omai (tw)
vfmsub231pd %ymm1,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm5,%ymm12,%ymm10 # iprod0 (tw)
vsubpd      %ymm8,%ymm0,%ymm1
vsubpd      %ymm10,%ymm4,%ymm5
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm10,%ymm4,%ymm4

# [r0 r1] [r2 r3] [r4 r5] [r6 r7]
# ymm0: r0,r1,r4,r5
# ymm8: r2,r3,r6,r7

2:
vmovapd     fft8neg(%rip),%xmm13
vmovupd     0x10(%rdx),%xmm12             # om: r,i
xorpd       %xmm12,%xmm13                 # om: r,-i
vinsertf128 $1, %xmm13, %ymm12, %ymm12    # om: r,i,r,-i
vshufpd     $3, %ymm12, %ymm12, %ymm13    # omi: i,i,r,r
vshufpd     $12,  %ymm12, %ymm12, %ymm12  # omr: r,r,-i,-i

vperm2f128 $0x31,%ymm1,%ymm0,%ymm8   # ymm8 contains re to mul (tw)
vperm2f128 $0x31,%ymm5,%ymm4,%ymm10   # ymm10 contains im to mul (tw)
vperm2f128 $0x20,%ymm1,%ymm0,%ymm0   # ymm0 contains re to add (tw)
vperm2f128 $0x20,%ymm5,%ymm4,%ymm2   # ymm2 contains im to add (tw)

vmulpd      %ymm10,%ymm13,%ymm4 # ia0.omi (tw)
vmulpd      %ymm8,%ymm13,%ymm6 # ra0.omi (tw)
vfmsub231pd %ymm8,%ymm12,%ymm4 # rprod0 (tw)
vfmadd231pd %ymm10,%ymm12,%ymm6 # iprod0 (tw)
vsubpd      %ymm4,%ymm0,%ymm8
vsubpd      %ymm6,%ymm2,%ymm10
vaddpd      %ymm4,%ymm0,%ymm0
vaddpd      %ymm6,%ymm2,%ymm2

vperm2f128 $0x31,%ymm10,%ymm2,%ymm5
vperm2f128 $0x20,%ymm10,%ymm2,%ymm4
vperm2f128 $0x31,%ymm8,%ymm0,%ymm1
vperm2f128 $0x20,%ymm8,%ymm0,%ymm0

3:
vmovupd    0x20(%rdx),%ymm12               # om:  r0,r1,i0,i1
vperm2f128 $0x01,%ymm12,%ymm12,%ymm13      # omi: i0,i1,r0,r1
vxorpd     fft8neg2(%rip),%ymm12,%ymm12    # omr: r0,r1,-i0,-i1

vunpckhpd %ymm1,%ymm0,%ymm8
vunpckhpd %ymm5,%ymm4,%ymm10
vunpcklpd %ymm1,%ymm0,%ymm0
vunpcklpd %ymm5,%ymm4,%ymm2

vmulpd      %ymm10,%ymm13,%ymm4 # ia0.omi (tw)
vmulpd      %ymm8,%ymm13,%ymm6 # ra0.omi (tw)
vfmsub231pd %ymm8,%ymm12,%ymm4 # rprod0 (tw)
vfmadd231pd %ymm10,%ymm12,%ymm6 # iprod0 (tw)
vsubpd      %ymm4,%ymm0,%ymm8
vsubpd      %ymm6,%ymm2,%ymm10
vaddpd      %ymm4,%ymm0,%ymm0
vaddpd      %ymm6,%ymm2,%ymm2

vunpckhpd %ymm8,%ymm0,%ymm1   # (0,4) -> (0,1)
vunpckhpd %ymm10,%ymm2,%ymm5    # (2,6) -> (2,3)
vunpcklpd %ymm8,%ymm0,%ymm0
vunpcklpd %ymm10,%ymm2,%ymm4

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
fft8neg: .quad 0x0, 0x8000000000000000

