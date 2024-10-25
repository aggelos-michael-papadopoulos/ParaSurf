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

#define	DEF_DIR(x)	DESTLIB#x

/*
 * SERVER_FILE contains a list of hosts that run dms servers (and are
 * binary-compatible with the host running dms)
 */
#define	SERVER_FILE	"dms_servers"
#define	DEF_SERVER_FILE	DEF_DIR(dms_servers)

/*
 * RADII_FILE contains a list of radii
 */
#define	RADII_FILE	"radii"
#define	DEF_RADII_FILE	DEF_DIR(radii)

/*
 * LOCK_FILE is the name of the file that a dms server tries to lock.
 * If the file exists and the lock fails, then the server will exit.
 */
#define	LOCK_FILE	DEF_DIR(lockfile)

/*
 * NICE_PRIORITY is the priority that a dms server sets itself to.
 */
#define	NICE_PRIORITY	15

/* 
 * The following #defines are for possible code options.  The best
 * combination for running on Suns is currently selected.  Your mileage
 * may vary with other (e.g. vectorizing) compilers.
 */
#define SORTED_NB	/* Neighbor list sorted by distance */
#undef	ONE_OCCLUDE	/* Only check occlusion after summing components */
#define KEEP_REAL	/* Keep only non-hidden probes */
#define RETRANS_NB	/* Retransmit shortened neighbor list to servers */
