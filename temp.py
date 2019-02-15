if A3[i, 0]*gamma3*(A2[0,0]*gamma1*A1[0,1] + A2[0,1]*gamma2*A1[1,1]) + A3[i, 1]*gamma4*(A2[1,0]*gamma1*A1[0,1] + A2[1,1]*gamma2*A1[1,1]) != 0:
    x = np.linspace(-10, 10, 1000)
    y = (-b3[i] - A3[i, 0]*gamma3*(A2[0, 0]*gamma1*(A1[0, 0]*x + b1[0]) + A2[0, 1]*gamma2*(A1[1, 0]*x + b1[1]) + b2[0])
                - A3[i, 1]*gamma4*(A2[1, 0]*gamma1*(A1[0, 0]*x + b1[0]) + A2[1, 1]*gamma2*(A1[1, 0]*x + b1[1]) + b2[1]))/ \
        (A3[i, 0]*gamma3*(A2[0,0]*gamma1*A1[0,1] + A2[0,1]*gamma2*A1[1,1]) + A3[i, 1]*gamma4*(A2[1,0]*gamma1*A1[0,1] + A2[1,1]*gamma2*A1[1,1]))
elif A3[i, 0]*gamma3*(A2[0,0]*gamma1*A1[0,0] + A2[0,1]*gamma2*A1[1,0]) + A3[i, 1]*gamma4*(A2[1,0]*gamma1*A1[0,0] + A2[1,1]*gamma2*A1[1,0]) != 0:
    y = np.linspace(-10, 10, 1000)
    x = (-b3[i] - A3[i, 0]*gamma3*(A2[0, 0]*gamma1*(A1[0, 1]*y + b1[0]) + A2[0, 1]*gamma2*(A1[1, 1]*y + b1[1]) + b2[0])
                - A3[i, 1]*gamma4*(A2[1, 0]*gamma1*(A1[0, 1]*y + b1[0]) + A2[1, 1]*gamma2*(A1[1, 1]*y + b1[1]) + b2[1]))/ \
        (A3[i, 0]*gamma3*(A2[0,0]*gamma1*A1[0,0] + A2[0,1]*gamma2*A1[1,0]) + A3[i, 1]*gamma4*(A2[1,0]*gamma1*A1[0,0] + A2[1,1]*gamma2*A1[1,0]))
