.p2align 4
.globl	reim_fftvec_addmul_fma
reim_fftvec_addmul_fma:
.globl	_reim_fftvec_addmul_fma
_reim_fftvec_addmul_fma:

	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15

	/* fftvec_addmul_fma(PRECOMP*, double* r, double* a, double* b) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of rre
	movq	%rdx, %r10         # r10: base of are
	movq	%rcx, %r12         # r12: base of bre
	leaq	(%r8,%rdi,8),%r9   # r9: base of rim
	leaq	(%r10,%rdi,8),%r11 # r11: base of aim
	leaq	(%r12,%rdi,8),%r13 # r13: base of bim
	movq	%r9,%r14           # r14: end of loop
1:
	vmovupd (%r8),%ymm8 /* rre */
	vmovupd (%r9),%ymm9 /* rim */
	vmovupd (%r10),%ymm0
	vmovupd (%r11),%ymm1
	vmovupd (%r12),%ymm2
	vmovupd (%r13),%ymm3
    vfmsub231pd	%ymm3,%ymm1,%ymm8 /* rre  = -rre + aim.bim */
    vfmsub231pd	%ymm2,%ymm0,%ymm8 /* rre  = -rre + are.bre */
	vfmadd231pd	%ymm3,%ymm0,%ymm9 /* rim  += are.bim */
	vfmadd231pd	%ymm2,%ymm1,%ymm9 /* rim  += aim.bre */
	vmovupd	%ymm8,(%r8)
	vmovupd	%ymm9,(%r9)
	/* end of loop */
	addq	$32,%r8
	addq	$32,%r9
	addq	$32,%r10
	addq	$32,%r11
	addq	$32,%r12
	addq	$32,%r13
	cmpq	%r14,%r8
	jb 1b

	vzeroall
	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12
	retq


.globl	reim_fftvec_mul_fma
reim_fftvec_mul_fma:
.globl	_reim_fftvec_mul_fma
_reim_fftvec_mul_fma:
	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15
	/* fftvec_addmul_fma(PRECOMP*, double* r, double* a, double* b) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of rre
	movq	%rdx, %r10         # r10: base of are
	movq	%rcx, %r12         # r12: base of bre
	leaq	(%r8,%rdi,8),%r9   # r9: base of rim
	leaq	(%r10,%rdi,8),%r11 # r11: base of aim
	leaq	(%r12,%rdi,8),%r13 # r13: base of bim
	movq	%r9,%r14           # r14: end of loop
0:
	vmovupd (%r10),%ymm8  # are
	vmovupd (%r11),%ymm9  # aim
	vmovupd (%r12),%ymm0 # bre
	vmovupd (%r13),%ymm1 # bim
    vmulpd	%ymm9,%ymm1,%ymm2 /* y2  = aim.bim */
    vmulpd	%ymm8,%ymm1,%ymm3 /* y3  = are.bim */
	vfmsub231pd	%ymm8,%ymm0,%ymm2 /* y2  = are.bre - y2 */
	vfmadd231pd	%ymm0,%ymm9,%ymm3 /* y3  = aim.bre + y3 */
	vmovupd	%ymm2,(%r8)
	vmovupd	%ymm3,(%r9)
	/* end of loop */
	addq	$32,%r8
	addq	$32,%r9
	addq	$32,%r10
	addq	$32,%r11
	addq	$32,%r12
	addq	$32,%r13
	cmpq	%r14,%r8
	jb 0b

	vzeroall
	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12
	retq
