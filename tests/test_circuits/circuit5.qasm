OPENQASM 2.0;
include "qelib1.inc";
qreg q6[4];
rx(pi/2) q6[1];
rx(pi) q6[2];
cx q6[2],q6[3];
cx q6[1],q6[2];
cx q6[0],q6[1];
