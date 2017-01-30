"use strict";

// import numpy as np
// 
// # sigmoid function
// def nonlin(x,deriv=False):
//     if(deriv==True):
//         return x*(1-x)
//     return 1/(1+np.exp(-x))

const nonlin = (a, deriv) => {
    const r = [];
    a.forEach((x) => {
        if (deriv) {
            r.push(x * (1 - x));
        } else {
            r.push(1 / (1 + Math.exp(0 - x)));
        }
    });
    return r;
};


//     
// # input dataset
// X = np.array([  [0,0,1],
//                 [0,1,1],
//                 [1,0,1],
//                 [1,1,1] ])

const X = [
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
];

//     
// # output dataset            
// y = np.array([[0,0,1,1]]).T
// 

const y = [
    [0],
    [0],
    [1],
    [1]
];

const genArray = (x, y, fn) => {
    const r = [];
    for (let i = 0; i < x; i++) {
        r[i] = r[i] || [];
        for (let j = 0; j < y; j++) {
            r[i][j] = fn(i, j);
        }
    }
    return r;
};

// # seed random numbers to make calculation
// # deterministic (just a good practice)
// np.random.seed(1)
// 

// # initialize weights randomly with mean 0
// syn0 = 2*np.random.random((3,1)) - 1
// 

let syn0 = genArray(3, 1, () => {
    return 2 * Math.random() - 1;
});

syn0 = [
    [0.5],
    [0],
    [-0.5]
];

syn0 = [
    [-0.16595599],
    [0.44064899],
    [-0.99977125],
];
// for iter in xrange(10000):
// 

const dot = (a1, a2) => {
    const r = [];
    a1.forEach((v1, i1) => {
        r[i1] = r[i1] || [];
        v1.forEach((v2, i2) => {
            a2[i2].forEach((v3, i3) => {
                r[i1][i3] = r[i1][i3] || 0;
                r[i1][i3] += a1[i1][i2] * a2[i2][i3];
            });
        });
    });
    return r;
};

let l1, l0, l1_error;

for (let iter = 0; iter < 1; iter++) {

    //     # forward propagation
    //     l0 = X
    //     l1 = nonlin(np.dot(l0,syn0))

    l0 = X;
    l1 = nonlin(dot(l0, syn0));

    //     # how much did we miss?
    //     l1_error = y - l1

    l1_error = y - l1;

    console.log(syn0);
    console.log(y);
    console.log(l1);

    //     # multiply how much we missed by the 
    //     # slope of the sigmoid at the values in l1
    //     l1_delta = l1_error * nonlin(l1,True)
    // 
    //     # update weights
    //     syn0 += np.dot(l0.T,l1_delta)

}

// 
// print "Output After Training:"
// print l1
//
