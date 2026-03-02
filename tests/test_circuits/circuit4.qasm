OPENQASM 2.0;
include "qelib1.inc";
qreg q5[4];
rx(3*pi/2) q5[3];
cx q5[3],q5[2];
cx q5[2],q5[1];
cx q5[0],q5[1];
