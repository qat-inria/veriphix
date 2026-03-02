OPENQASM 2.0;
include "qelib1.inc";
qreg q9[4];
rx(pi) q9[3];
cx q9[3],q9[2];
cx q9[2],q9[1];
cx q9[0],q9[1];
