! Copyright (C) 2020 Jonas A. Finkler
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

! Efficient implementation of the Kalman filter algorithm to train a neural network
! written by Jonas Finkler

module kalmanFilter
    implicit none

contains

    ! As P is diagonal, only the lower half is used
    subroutine updatekalman(lam0, lam, err, d, w, J, P)
        real(8), intent(in) :: lam0
        real(8), intent(inout) :: lam
        real(8), intent(in) :: err
        integer, intent(in) :: d ! number of parameters
        real(8), intent(inout) :: w(d) ! parameters
        real(8), intent(in) :: J(d) ! derr/dw
        real(8), intent(inout) :: P(d,d)

        real(8) :: PJ(d)!, JP(d)
        real(8) :: kk

        ! PJ = P @ J
        call matMulVec(d, P, J, PJ)
        ! prefactor for K  | K = kk * PJ
        kk = 1.d0 / (lam + sum(J * PJ))
        ! P = P - kk * PJ @ PJ^T
        call matMinusVecMulVecT(d, P, PJ, kk)
        ! P *= 1 / lam
        P = P * (1.d0 / lam)
        !w = w - K * err
        w(:) = w(:) - PJ(:) * (err * kk)
        ! update lam
        lam = lam * lam0 + 1.d0 - lam0

    end subroutine

!   ---------- subroutines to perform matrix operations using BLAS ---------- !

    ! M = M - alpha * x * x^t
    subroutine matMinusVecMulVecT(d, M, x, alpha)
        integer, intent(in) :: d
        real(8), intent(inout) :: M(d,d)
        real(8), intent(in) :: x(d)
        real(8), intent(in) :: alpha

        call DSYR('L', d, -alpha, x, 1, M, d)

    end subroutine

    ! v = A * x
    subroutine MatMulVec(d, A, x, v)
        integer, intent(in) :: d
        real(8), intent(in) :: A(d,d), x(d)
        real(8), intent(out) :: v(d)

        call DSYMV('L', d, 1.d0, A, d, x, 1, 0.d0, v, 1)

    end subroutine

end module
