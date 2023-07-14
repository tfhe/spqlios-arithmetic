.p2align 4

.globl	reim4_from_cplx_fma
.type	reim4_from_cplx_fma, @function
reim4_from_cplx_fma:
.globl	_reim4_from_cplx_fma
_reim4_from_cplx_fma:

	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15

	/* from_cplx_fma(PRECOMP*, double* r, double* a) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of res
	movq	%rdx, %r10         # r8: base of in
	movq	%rdi,%r14          # r14: end of loop
	shlq    $4,%r14
	addq    %r8,%r14
1:
	vmovupd (%r10),%ymm8 /* a */
	vmovupd 0x20(%r10),%ymm9 /* b */
	vunpcklpd %ymm9,%ymm8,%ymm10
    vunpckhpd %ymm9,%ymm8,%ymm11
	vmovupd	%ymm10,(%r8)
	vmovupd	%ymm11,0x20(%r8)
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
.size	reim4_from_cplx_fma, .-reim4_from_cplx_fma

.globl	reim4_to_cplx_fma
.type	reim4_to_cplx_fma, @function
reim4_to_cplx_fma:
.globl	_reim4_to_cplx_fma
_reim4_to_cplx_fma:

	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15

	/* from_cplx_fma(PRECOMP*, double* r, double* a) */
	movslq  8(%rdi),%rdi       # rdi: m
	movq	%rsi, %r8          # r8: base of res
	movq	%rdx, %r10         # r8: base of in
	movq	%rdi,%r14          # r14: end of loop
	shlq    $4,%r14
	addq    %r8,%r14
1:
	vmovupd (%r10),%ymm8 /* a */
	vmovupd 0x20(%r10),%ymm9 /* b */
	vunpcklpd %ymm9,%ymm8,%ymm10
    vunpckhpd %ymm9,%ymm8,%ymm11
	vmovupd	%ymm10,(%r8)
	vmovupd	%ymm11,0x20(%r8)
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
.size	reim4_to_cplx_fma, .-reim4_to_cplx_fma


.section .note.GNU-stack,"",@progbits
