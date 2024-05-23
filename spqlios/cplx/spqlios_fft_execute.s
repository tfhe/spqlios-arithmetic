.globl cplx_fftvec_innerprod
cplx_fftvec_innerprod:
  jmp *(%rdi)
.globl cplx_from_znx32
cplx_from_znx32:
  jmp *(%rdi)
.globl cplx_from_tnx32
cplx_from_tnx32:
  jmp *(%rdi)
.globl cplx_to_tnx32
cplx_to_tnx32:
  jmp *(%rdi)
.globl cplx_fftvec_mul
cplx_fftvec_mul:
  jmp *(%rdi)
.size	cplx_fftvec_mul, .-cplx_fftvec_mul
.globl cplx_fftvec_addmul
cplx_fftvec_addmul:
  jmp *(%rdi)
.size	cplx_fftvec_addmul, .-cplx_fftvec_addmul

.section	.note.GNU-stack,"",@progbits
