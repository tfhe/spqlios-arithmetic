.p2align 4
.globl	reim4_fftvec_addmul_fma
reim4_fftvec_addmul_fma:
.globl	_reim4_fftvec_addmul_fma
_reim4_fftvec_addmul_fma:

	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15

	/* fftvec_addmul_fma(PRECOMP*, double* r, double* a, double* b) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of rre
	movq	%rdx, %r10         # r10: base of are
	movq	%rcx, %r12         # r12: base of bre
	movq	%rdi,%r14          # r14: end of loop
	shlq    $4,%r14
	addq    %r8,%r14
1:
	vmovupd (%r8),%ymm8 /* rre */
	vmovupd 0x20(%r8),%ymm9 /* rim */
	vmovupd (%r10),%ymm0
	vmovupd 0x20(%r10),%ymm1
	vmovupd (%r12),%ymm2
	vmovupd 0x20(%r12),%ymm3
    vfmsub231pd	%ymm3,%ymm1,%ymm8 /* rre  = -rre + aim.bim */
    vfmsub231pd	%ymm2,%ymm0,%ymm8 /* rre  = -rre + are.bre */
	vfmadd231pd	%ymm3,%ymm0,%ymm9 /* rim  += are.bim */
	vfmadd231pd	%ymm2,%ymm1,%ymm9 /* rim  += aim.bre */
	vmovupd	%ymm8,(%r8)
	vmovupd	%ymm9,0x20(%r8)
	/* end of loop */
	addq	$64,%r8
	addq	$64,%r10
	addq	$64,%r12
	cmpq	%r14,%r8
	jb 1b

	vzeroall
	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12
	retq


.globl	reim4_fftvec_mul_fma
reim4_fftvec_mul_fma:
.globl	_reim4_fftvec_mul_fma
_reim4_fftvec_mul_fma:
	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15
	/* fftvec_addmul_fma(PRECOMP*, double* r, double* a, double* b) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of rre
	movq	%rdx, %r10         # r10: base of are
	movq	%rcx, %r12         # r12: base of bre
	movq    %rdi, %r14
	shlq    $4, %r14
	addq	%r8,%r14           # r14: end of loop
0:
	vmovupd (%r10),%ymm8  # are
	vmovupd 0x20(%r10),%ymm9  # aim
	vmovupd (%r12),%ymm0 # bre
	vmovupd 0x20(%r12),%ymm1 # bim
    vmulpd	%ymm9,%ymm1,%ymm2 /* y2  = aim.bim */
    vmulpd	%ymm8,%ymm1,%ymm3 /* y3  = are.bim */
	vfmsub231pd	%ymm8,%ymm0,%ymm2 /* y2  = are.bre - y2 */
	vfmadd231pd	%ymm0,%ymm9,%ymm3 /* y3  = aim.bre + y3 */
	vmovupd	%ymm2,(%r8)
	vmovupd	%ymm3,0x20(%r8)
	/* end of loop */
	addq	$64,%r8
	addq	$64,%r10
	addq	$64,%r12
	cmpq	%r14,%r8
	jb 0b

	vzeroall
	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12
	retq
