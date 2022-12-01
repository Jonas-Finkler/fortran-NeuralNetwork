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

program nnTest
    use precision
    use neuralNetworks
    use kalmanFilter
    use util
    implicit none

    ! nn architecture
    integer, parameter :: nInp = 1
    integer, parameter :: nOut = 1
    integer, parameter :: nLayers = 3
    integer, parameter :: nNodes(nLayers) = [10, 20, 1]
    character, parameter :: act(nLayers) = ['t', 't', 'l']

    ! kalman parameter
    real(dp), parameter :: lam0 = 0.9987

    ! training and testing data
    integer, parameter :: nTrain = 1000
    integer, parameter :: nTest = 1000
    real(dp) :: x(nTrain), y(nTrain), yy(nTrain), err
    real(dp) :: xT(nTest), yT(nTest), yyT(nTest)
    integer :: i, j, epoch, order(nTrain)

    type(neuralNetwork) :: nn
    real(dp) :: inp(nInp), out1(nOut), out2(nOut), dOut(nInp), ones(nOut)
    real(dp) :: dx(nInp)
    real(dp), allocatable :: params(:), dParams(:), dxParams(:), P(:,:)
    real(dp) :: lam

    if (nNodes(nLayers) /= nOut) stop 'wrong nOut'


    ! initialize NN
    call nn_new(nn, nInp, nLayers, nNodes, act)

    print*, 'nParams=', nn%nParams

    ! allocate arrays
    allocate(params(nn%nParams))
    allocate(dParams(nn%nParams))
    allocate(dxParams(nn%nParams))

    ! some tests for the NN derivatives
    if (.true.) then
        call random_number(dx)
        dx = dx / sqrt(sum(dx**2)) * 1.e-4_dp

        ones = 1._dp

        call random_number(inp)

        call nn_forward(nn, inp, out1)

        ! dX / dOut = 1 (ones)
        call nn_backward(nn, ones, dOut)

        inp = inp + dx
        call nn_forward(nn, inp, out2)

        ! check nn derivative with finitie difference
        print*, sum(dOut*dx) / sqrt(sum(dx**2)), '?=', sum(out2 - out1) / sqrt(sum(dx**2))

        call random_number(dxParams)
        dxParams = dxParams / sqrt(sum(dxParams**2)) * 1.e-5_dp

        call nn_forward(nn, inp, out1)
        call nn_backward(nn, ones, dOut)
        call nn_getdParams(nn, dParams)

        call nn_getParams(nn, params)
        params = params + dxParams
        call nn_setParams(nn, params)

        call nn_forward(nn, inp, out2)

        ! check nn derivative of weights with finitie difference
        print*, sum(dParams*dxParams) / sqrt(sum(dParams**2)), '?=', sum(out2 - out1) / sqrt(sum(dParams**2))
        print*, " "

    end if



    ! set up some training data for the NN
    call random_number(x)
    x = 2._dp * x - 1._dp
    do i=1,nTrain
        ! just some function that we can fit
        y(i) = sin(20._dp * x(i)) * x(i)
    end do

    do i=1,nTest
        xT(i) = real(i, dp) / real(nTest, dp) * 2._dp - 1.0_dp
        yT(i) = sin(20._dp * xT(i)) * xT(i)
    end do

    do i=1,nTrain
        order(i) = i
    end do

    ! initialize Kalman P matrix
    allocate(P(nn%nParams, nn%nParams))
    P = 0._dp
    do i=1,nn%nParams
        P(i,i) = 1.e5_dp
    end do

    ! kalman lambda
    lam = 0.99


    do epoch=1,1000
        call shuffle(nTrain, order)

        do i=1,nTrain

            inp(1) = x(order(i))

            call nn_forward(nn, inp, out1)
            err = out1(1) - y(order(i))
            call nn_backward(nn, [1._dp], dOut)

            call nn_getdParams(nn, dParams)
            call nn_getParams(nn, params)
            call updatekalman(lam0, lam, err, nn%nParams, params, dParams, P)
            call nn_setParams(nn, params)

        end do

        ! calculate error using the batch mode
        call nn_forwardBatch(nn, nTrain, x, yy)
        call nn_forwardBatch(nn, nTest, xT, yyT)
        print*, 'RMSE (train, test) = ', sqrt(sum((y - yy)**2) / nTrain), sqrt(sum((yT - yyT)**2) / nTest)


        ! write the result to a file every once in a while for plotting
        if(modulo(epoch,10) == 0) then
            print*, 'prediction'
            open(45, file='out.txt')
            do j=1,nTest
                write(45,*) xT(j), yT(j), yyT(j)
            end do
            close(45)
        end if

    end do



end program nnTest
