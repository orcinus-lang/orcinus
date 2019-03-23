/*******************************************************************************
 *     Copyright (c) 2019 Vasiliy Sheredeko vasiliy@sheredeko.me
 *         All Rights Reserved
 *
 *     THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF Vasiliy Sheredeko
 *     The copyright notice above does not evidence any
 *     actual or intended publication of such source code. The
 *     intellectual and technical concepts contained herein are proprietary to
 *     Vasiliy Sheredeko are protected by trade secret or copyright law.
 *     Dissemination of this information or reproduction of this material is
 *     strictly forbidden unless prior written permission is obtained from
 *     Vasiliy Sheredeko.
 ******************************************************************************/
#pragma once

#include "types.h"

/// Create initial wire
orx_wire_t* orx_wire_initial();

/// Create initial wire
orx_wire_t* orx_wire_create(orx_wire_func func, void* ptr);

/// Register wire in processor
void orx_wire_register(orx_wire_t* wire, orx_processor_t* processor);

/// Transfer execution to wire
void orx_wire_transfer(orx_wire_t* current_wire, orx_wire_t* next_wire);
