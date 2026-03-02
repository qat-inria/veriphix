OPENQASM 2.0;
include "qelib1.inc";
qreg q3[4];
rx(3*pi/2) q3[3];
cx q3[3],q3[2];
cx q3[2],q3[1];
cx q3[1],q3[0];
