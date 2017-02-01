// node app/neural-network-test.js
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

const operate = (a1, a2, fn) => {
    const r = [];
    a1.forEach((v, i) => {
        r[i] = [fn(v, a2[i])];
    });
    return r;
};

const matrixOperate = (fn) => {
    return (a1, a2) => {
        const r = [];
        a1.forEach((v, i) => {
            r[i] = [fn(v, a2[i])];
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


const operate2 = (a1, a2, fn) => {
    const r = [];
    a1.forEach((v1, i) => {
        v1.forEach((v2, j) => {
            r[i] = r[i] || [];
            r[i][j] = [fn(v2, a2[i][j])];
        });
    });
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

const transpose = (a) => {
    const r = [];
    a.forEach((v1, i) => {
        // console.log("xxxx", typeof v1);
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

let l1, l0, l1_error, l1_delta, temp;

for (let iter = 0; iter < 1; iter++) {

    //     # forward propagation
    //     l0 = X
    //     l1 = nonlin(np.dot(l0,syn0))

    l0 = X;
    l1 = nonlin(dot(l0, syn0));

    console.log(l0);
    console.log(syn0);
    console.log(dot(l0, syn0));
    console.log(nonlin(dot(l0, syn0)));

    //     # how much did we miss?
    //     l1_error = y - l1

    l1_error = operate(y, l1, (a, b) => {
        return (a - b);
    });

    //     # multiply how much we missed by the 
    //     # slope of the sigmoid at the values in l1
    //     l1_delta = l1_error * nonlin(l1,True)

    l1_delta = operate(l1_error, nonlin(l1, true), (a, b) => {
        return (a * b);
    });

    //     # update weights
    //     syn0 += np.dot(l0.T,l1_delta)

    //console.log(transpose(l0));
    //console.log(l1_delta);
    //temp = dot(transpose(l0), l1_delta);
    //console.log(temp);

    // console.log(syn0);
    syn0 = operate(syn0, dot(transpose(l0), l1_delta), (a, b) => {
        return (a + b);
    });


    // console.log(syn0);
    // console.log(l1_error);
    //console.log(transpose(l0));
    //console.log(l1_delta);
    //console.log(dot(transpose(l0), l1_delta));
    //console.log(syn0);
}

// 
// print "Output After Training:"
// print l1
//
