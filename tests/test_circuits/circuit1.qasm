OPENQASM 2.0;
include "qelib1.inc";
qreg q2[4];
rz(3*pi/4) q2[1];
rx(pi/2) q2[3];
cx q2[3],q2[2];
cx q2[1],q2[2];
cx q2[0],q2[1];
