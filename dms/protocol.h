/*

Copyright (c) <2002> The Regents of the University of California.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions, and the following disclaimer.
  2. Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions, and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
  3. Redistributions must acknowledge that this software was
     originally developed by the UCSF Computer Graphics Laboratory
     under support by the NIH National Center for Research Resources,
     grant P41-RR01081.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */
#define	PI_COMMAND	"paramdata radius %lf, density %lf, wantnormal %d\n"
#define	PI_RESPONSE	"received paramdata\n"

#define	AI_COMMAND	"atomdata %d\n"
#define	AI_RESPONSE	"received atomdata\n"

#define	NI_COMMAND	"neighbordata %d\n"
#define	NI_RESPONSE	"received neighbordata\n"

#define	PRI_COMMAND	"probedata %d\n"
#define	PRI_RESPONSE	"received probedata\n"

#define	C_COMMAND	"contacts %d\n"
#define	C_RESPONSE	"contactdata atom %d, %d neighbors\n"

#define	P_COMMAND	"probes %d %d\n"
#define	P_BADNB		"badneighbor %d %d\n"
#define	P_RESPONSE	"probedata %d probes\n"

#define	CS_COMMAND	"csurface %d\n"
#define	TS_COMMAND	"tsurface %d %d\n"
#define	PS_COMMAND	"psurface %d\n"

#define	SURFACE_END	"surfacedata end\n"
#define	SURFACE_RESPONSE \
	"surfacedata atom %d, %d points, type %d, area %lf, havenormal %d\n"
