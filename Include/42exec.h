/*    This file is distributed with 42,                               */
/*    the (mostly harmless) spacecraft dynamics simulation            */
/*    created by Eric Stoneking of NASA Goddard Space Flight Center   */

/*    Copyright 2010 United States Government                         */
/*    as represented by the Administrator                             */
/*    of the National Aeronautics and Space Administration.           */

/*    No copyright is claimed in the United States                    */
/*    under Title 17, U.S. Code.                                      */

/*    All Other Rights Reserved.                                      */

#include "42types.h"

#ifndef __42_EXEC_H__
#define __42_EXEC_H__

void ReportProgress(void);
void ManageFlags(void);
long AdvanceTime(void);
void UpdateScBoundingBox(struct SCType *S);
void ManageBoundingBoxes(void);
void ZeroFrcTrq(void);
long SimStep(const char *installedModelPath);

#endif



