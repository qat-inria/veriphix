OPENQASM 2.0;
include "qelib1.inc";
qreg q1[4];
cx q1[3],q1[2];
rz(pi) q1[2];
cx q1[1],q1[2];
cx q1[1],q1[0];
