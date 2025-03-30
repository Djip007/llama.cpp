#pragma once

#include "ggml-impl.h"
#include "ggml-hip.h"
#include "tools.h"

// choix de la version a builer...

//#define BLOC_V1    // mieux? N=[23-47]  => Voir comment faire "mieux" dans cette bande.
//#define BLOC_V2
//#define BLOC_V3
//#define BLOC_V4  // OK N=[1-22] N=[48...]
//#define BLOC_V5  // avec fp16 en plus ?
//#define BLOC_V6 
//#define BLOC_V7   // load vectorisé
//#define BLOC_V8
#define BLOC_V9   // lood A/B separement

#ifdef BLOC_V1
#include "mulmat-bf16bloc_V1.h"
#endif
#ifdef BLOC_V2
#include "mulmat-bf16bloc_V2.h"
#endif
#ifdef BLOC_V3
#include "mulmat-bf16bloc_V3.h"
#endif
#ifdef BLOC_V4
#include "mulmat-bf16bloc_V4.h"
#endif
#ifdef BLOC_V5
#include "mulmat-bf16bloc_V5.h"
#endif
#ifdef BLOC_V6
#include "mulmat-bf16bloc_V6.h"
#endif
#ifdef BLOC_V7
#include "mulmat-bf16bloc_V7.h"
#endif
#ifdef BLOC_V8
#include "mulmat-bf16bloc_V8.h"
#endif
#ifdef BLOC_V9
#include "mulmat-bf16bloc_V9.h"
#endif
