OPENQASM 2.0;
include "qelib1.inc";
qreg q8[4];
rx(3*pi/4) q8[3];
cx q8[3],q8[2];
cx q8[1],q8[2];
cx q8[1],q8[0];
