// $ node app/neural-network-test.js
// https://repl.it/F5Mw/3
// https://github.com/lucamug/simple-neural-network-in-javascript
"use strict";
const matrixOperate = (fn) => {
    return (a1, a2) => {
        // console.log('xxxx', fn, a1, a2);
        const r = [];
        a1.forEach((v, i) => {
            // console.log("iiiiii", v[0], a2[i][0]);
            r[i] = [fn(v[0], a2[i][0])];
        });
        return r;
    };
};
const matrixAdd = matrixOperate((a, b) => {
    return a + b;
});
const matrixSub = matrixOperate((a, b) => {
    return a - b;
});
const matrixMul = matrixOperate((a, b) => {
    return a * b;
});
const transpose = (a) => {
    const r = [];
    a.forEach((v1, i) => {
        v1.forEach((v2, j) => {
            r[j] = r[j] || [];
            r[j][i] = v2;
        });
    });
    return r;
};
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
let l1, l0, l1_error, l1_delta;

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
            r.push([x * (1 - x)]);
        } else {
            r.push([1 / (1 + Math.exp(-x))]);
        }
    });
    return r;
};

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

// # output dataset            
// y = np.array([[0,0,1,1]]).T

const y = transpose([
    [0, 0, 1, 1]
]);

// # seed random numbers to make calculation
// # deterministic (just a good practice)
// np.random.seed(1)

// # initialize weights randomly with mean 0
// syn0 = 2*np.random.random((3,1)) - 1 

let syn0 = [
    [-0.16595599],
    [0.44064899],
    [-0.99977125],
];

// for iter in xrange(10000):

for (let iter = 0; iter < 10000; iter++) {

    //     # forward propagation
    //     l0 = X
    //     l1 = nonlin(np.dot(l0,syn0))

    l0 = X;
    l1 = nonlin(dot(l0, syn0));

    //     # how much did we miss?
    //     l1_error = y - l1

    l1_error = matrixSub(y, l1);

    //     # multiply how much we missed by the 
    //     # slope of the sigmoid at the values in l1
    //     l1_delta = l1_error * nonlin(l1,True)

    l1_delta = matrixMul(l1_error, nonlin(l1, true));

    //     # update weights
    //     syn0 += np.dot(l0.T,l1_delta)

    syn0 = matrixAdd(syn0, dot(transpose(l0), l1_delta));
}

// print "Output After Training:"
// print l1

console.log("Output After Training:");
console.log(l1);
