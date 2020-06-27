

module util
  use precision
  implicit none

contains

    ! knuth shuffle
    subroutine shuffle(n, a)
        integer, intent(in) :: n
        integer, intent(inout) :: a(n)
        integer :: i, randpos, temp
        real(dp) :: r

        do i = n, 2, -1
            call random_number(r)
            randpos = int(r * i) + 1
            temp = a(randpos)
            a(randpos) = a(i)
            a(i) = temp
        end do

    end subroutine shuffle



end module
