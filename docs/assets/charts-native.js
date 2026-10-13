/*!
 * charts-native.js - LiEqualizer spectral diagnostics, rendered as native inline SVG.
 *
 * No build step, no framework, no bundler, no CDN. Vanilla ES5 + inline SVG only.
 * Public API: window.LiEQCharts (see charts-native.README.md).
 *
 * Data: raw float64 values exported by results/S4/APPX_D_native_web_data_20261013/
 * export_web_data.py, embedded verbatim below (shortest round-trip literals, so the
 * browser parses back bit-identical doubles). Colour in these six figures encodes the
 * DATASET; dense vs sparse is encoded by fill and line style, never by colour.
 */
;(function (global) {
  'use strict';

  /* ------------------------------------------------------------------ data */

  var DATA = {
    meta: {
      source: "results/S4/APPX_D_native_web_data_20261013/fig_native_data.json",
      generated_by: "export_web_data.py (S4)",
      precision: "all values are raw float64 from the same arrays the figures were drawn from; display formats are given per figure",
      run: "APPX_D_native_finalmap_20261012 (S4, 2026-10-12 21:02 → 10-13 00:17 SGT, exit 0)",
      frames_per_dataset: 200,
      frame_selection: "seed-0 stratified sampling prefix (same 200 frames as the 9-23 run)",
      dense: "LiDiff refine completion, uniformly subsampled without replacement to 500,000 points",
      sparse: "real raw LiDAR scan (no ray casting)",
      label_mapping: "final coarse-label mapping, finalized 2026-09-22 (source of truth coarse_mapping.py md5 704fabec)",
      class_convention: "7 classes (class 0 Other excluded)",
      spectrum: "KPM/Chebyshev on the kNN graph Laplacian, lambda axis [0, 2]",
      dataset_order: ["SemanticKITTI", "Waymo", "nuScenes"],
      summary_md5: {
        semantickitti: "35b0185db835b83df0156061ba5c1689",
        waymo: "c11aff1c69c4f58a8036903518b4ca5f",
        nuscenes: "f7ac07d9361900e536a315fe0f2fec56"
      }
    },
    basis: {
      qprofiles: {
        statistic: "per-frame normalized profile, then MEAN over the 200 frames (summary q_d_mean / q_s_mean)",
        note: "x axis is diffusion steps t, not b0"
      },
      rolloff: {
        statistic: "left: per-frame λ_85 rolloff, MEDIAN over frames; right: per-frame λ≥0.6 energy share, MEAN over frames",
        note: "HFR^{sig} (per-frame sparse/dense share ratio, median) is NOT drawn here; it is in RESULTS.md table A"
      },
      retention: {
        statistic: "bars = median, whiskers = IQR, diamonds = median",
        note: "diamond n can be smaller than the printed n (needs the class on BOTH dense and sparse sides); see RESULTS.md table B"
      },
      heatmap: {
        statistic: "per-frame band-energy profile P (normalized to sum 1), gap G = P_dense − P_sparse, MEAN over frames",
        note: "class rows use only frames where the class exists on both sides (n per row)"
      },
      b0trend: {
        statistic: "R^{(c)}_{b0}: per-frame diffusion HF retention, MEDIAN over frames, b0 ∈ {2,8,16,32}",
        note: null
      },
      bands: {
        statistic: "per-frame normalized band-energy profile, MEAN over frames (summary band_d_mean / band_s_mean)",
        note: null
      }
    },
    datasets: {
      semantickitti: {
        display: "SemanticKITTI",
        short: "SK",
        color: "#4477AA"
      },
      waymo: {
        display: "Waymo",
        short: "WY",
        color: "#EE6677"
      },
      nuscenes: {
        display: "nuScenes",
        short: "NS",
        color: "#228833"
      }
    },
    classColors: {
      "Driveable Ground": "#1f77b4",
      Structure: "#ff7f0e",
      Vehicle: "#2ca02c",
      Nature: "#d62728",
      "Living Being": "#9467bd",
      Objects: "#8c564b",
      "Other Ground": "#e377c2"
    },
    qprofiles: {
      x: [1, 4, 16, 32],
      xname: "diffusion steps t",
      yname: "normalized response energy q̄_t",
      yunit: "fraction; per frame sums to 1 over the 4 t values",
      panels: [
        {
          signal: "density",
          title: "density",
          series: [
            {
              name: "SemanticKITTI dense",
              ds: "semantickitti",
              side: "dense",
              color: "#4477AA",
              line: "solid",
              marker: "circle",
              v: [0.31645383141670247, 0.27128639674858573, 0.217523494594566, 0.19473627723980225]
            },
            {
              name: "SemanticKITTI sparse",
              ds: "semantickitti",
              side: "sparse",
              color: "#4477AA",
              line: "dashed",
              marker: "square",
              v: [0.26154472576143517, 0.25324395564382546, 0.24459345369782481, 0.24061786489664722],
              drd: 0.003116323606689007
            },
            {
              name: "Waymo dense",
              ds: "waymo",
              side: "dense",
              color: "#EE6677",
              line: "solid",
              marker: "circle",
              v: [0.31237013280500153, 0.269407752251028, 0.2194440158281022, 0.19877809911552963]
            },
            {
              name: "Waymo sparse",
              ds: "waymo",
              side: "sparse",
              color: "#EE6677",
              line: "dashed",
              marker: "square",
              v: [0.26197401497243444, 0.2538443285986721, 0.24462209705127244, 0.2395595593773539],
              drd: 0.002545157066236804
            },
            {
              name: "nuScenes dense",
              ds: "nuscenes",
              side: "dense",
              color: "#228833",
              line: "solid",
              marker: "circle",
              v: [0.2940834552654735, 0.2635582283201718, 0.22851492262632842, 0.2138433937877135]
            },
            {
              name: "nuScenes sparse",
              ds: "nuscenes",
              side: "sparse",
              color: "#228833",
              line: "dashed",
              marker: "square",
              v: [0.26279132798322397, 0.2544154278993333, 0.24421616756058984, 0.2385770765565825],
              drd: 0.0012627322548203507
            }
          ]
        },
        {
          signal: "normal_z",
          title: "normal_z",
          series: [
            {
              name: "SemanticKITTI dense",
              ds: "semantickitti",
              side: "dense",
              color: "#4477AA",
              line: "solid",
              marker: "circle",
              v: [0.3251632471549242, 0.2536059196915666, 0.2158653589417989, 0.20536547421132476]
            },
            {
              name: "SemanticKITTI sparse",
              ds: "semantickitti",
              side: "sparse",
              color: "#4477AA",
              line: "dashed",
              marker: "square",
              v: [0.2966848511571888, 0.2565554043544575, 0.22871101410777328, 0.21804873038025122],
              drd: 0.0005919851834058792
            },
            {
              name: "Waymo dense",
              ds: "waymo",
              side: "dense",
              color: "#EE6677",
              line: "solid",
              marker: "circle",
              v: [0.29952439073267606, 0.2534591496567255, 0.22774180437379926, 0.2192746552364607]
            },
            {
              name: "Waymo sparse",
              ds: "waymo",
              side: "sparse",
              color: "#EE6677",
              line: "dashed",
              marker: "square",
              v: [0.2772353550377954, 0.2546448865637753, 0.23770786329119573, 0.2304118951069386],
              drd: 0.0003709277919054056
            },
            {
              name: "nuScenes dense",
              ds: "nuscenes",
              side: "dense",
              color: "#228833",
              line: "solid",
              marker: "circle",
              v: [0.3062019358942397, 0.25111820363273607, 0.2245492707064086, 0.21813058976626173]
            },
            {
              name: "nuScenes sparse",
              ds: "nuscenes",
              side: "sparse",
              color: "#228833",
              line: "dashed",
              marker: "square",
              v: [0.2780336847805513, 0.2568479256750509, 0.23713040893514134, 0.2279879806089624],
              drd: 0.0006858784001949964
            }
          ]
        },
        {
          signal: "surface_variation",
          title: "surface_variation",
          series: [
            {
              name: "SemanticKITTI dense",
              ds: "semantickitti",
              side: "dense",
              color: "#4477AA",
              line: "solid",
              marker: "circle",
              v: [0.29198298236248754, 0.2557255689343099, 0.23065693999286085, 0.22163450871002027]
            },
            {
              name: "SemanticKITTI sparse",
              ds: "semantickitti",
              side: "sparse",
              color: "#4477AA",
              line: "dashed",
              marker: "square",
              v: [0.28175532282023985, 0.25648245390967284, 0.2353556048911218, 0.22640661837866488],
              drd: 0.00010248570256070295
            },
            {
              name: "Waymo dense",
              ds: "waymo",
              side: "dense",
              color: "#EE6677",
              line: "solid",
              marker: "circle",
              v: [0.2870358549449932, 0.2554858874615807, 0.23289214594499405, 0.22458611164811984]
            },
            {
              name: "Waymo sparse",
              ds: "waymo",
              side: "sparse",
              color: "#EE6677",
              line: "dashed",
              marker: "square",
              v: [0.28220192086997575, 0.2557949821792737, 0.23518572442252367, 0.22681737252792405],
              drd: 5.290519058558877e-05
            },
            {
              name: "nuScenes dense",
              ds: "nuscenes",
              side: "dense",
              color: "#228833",
              line: "solid",
              marker: "circle",
              v: [0.2870671858996355, 0.25508809673158106, 0.232764495833602, 0.22508022153486876]
            },
            {
              name: "nuScenes sparse",
              ds: "nuscenes",
              side: "sparse",
              color: "#228833",
              line: "dashed",
              marker: "square",
              v: [0.28378223406727265, 0.2583377828949464, 0.2345307283505524, 0.22334925468692543],
              drd: 8.815635428656921e-05
            }
          ]
        }
      ]
    },
    rolloff: {
      encoding: {
        color: "dataset (orig_color)",
        dense: "solid fill, alpha 0.92, edge #555555",
        sparse: "same dataset color, alpha 0.38, hatch '///'",
        bar_label_format: "{:.3g}, rotated 90°"
      },
      panels: [
        {
          key: "left",
          title: "Spectral rolloff (θ=0.85): dense vs sparse",
          xname: "global signal",
          signals: ["density", "normal_z", "surface_variation"],
          yname: "spectral rolloff λ_85 (median over frames)",
          yunit: "graph frequency λ ∈ [0,2]",
          yscale: "log; upper limit ×2.2 headroom",
          series: [
            {
              signal: "density",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.4547263711542749,
              sparse: 0.009654577677756546
            },
            {
              signal: "density",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.4519630852794938,
              sparse: 0.009777725781239305
            },
            {
              signal: "density",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.3286567109793179,
              sparse: 0.009380948644248568
            },
            {
              signal: "normal_z",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.9940711035805199,
              sparse: 0.7671916578666342
            },
            {
              signal: "normal_z",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.9027350528793674,
              sparse: 0.15174959386970022
            },
            {
              signal: "normal_z",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.9661595486806511,
              sparse: 0.18938170350130962
            },
            {
              signal: "surface_variation",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.6880297155677981,
              sparse: 0.2893059731852171
            },
            {
              signal: "surface_variation",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.5565736787103137,
              sparse: 0.3613184188090939
            },
            {
              signal: "surface_variation",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.575505468650728,
              sparse: 0.26314802533678183
            }
          ]
        },
        {
          key: "right",
          title: "True high-frequency share (λ ≥ 0.6): dense vs sparse",
          xname: "global signal",
          signals: ["density", "normal_z", "surface_variation"],
          yname: "true-HF share P(λ ≥ 0.6) (mean over frames)",
          yunit: "energy fraction",
          yscale: "log; upper limit ×2.2 headroom",
          series: [
            {
              signal: "density",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.10638959717728934,
              sparse: 0.03354235289481664
            },
            {
              signal: "density",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.10429984744554999,
              sparse: 0.031966260576369955
            },
            {
              signal: "density",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.08392514015456469,
              sparse: 0.0360153232010616
            },
            {
              signal: "normal_z",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.3078646405863287,
              sparse: 0.1796368787672763
            },
            {
              signal: "normal_z",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.22598893536248546,
              sparse: 0.10981263900641272
            },
            {
              signal: "normal_z",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.26843683038145066,
              sparse: 0.08931186241862424
            },
            {
              signal: "surface_variation",
              ds: "semantickitti",
              color: "#4477AA",
              dense: 0.16641515951361385,
              sparse: 0.11201620440462629
            },
            {
              signal: "surface_variation",
              ds: "waymo",
              color: "#EE6677",
              dense: 0.14623953971557666,
              sparse: 0.1207015321329235
            },
            {
              signal: "surface_variation",
              ds: "nuscenes",
              color: "#228833",
              dense: 0.1488838259357556,
              sparse: 0.10620907801417796
            }
          ]
        }
      ]
    },
    retention: {
      classes: ["Driveable Ground", "Structure", "Vehicle", "Nature", "Living Being", "Objects", "Other Ground"],
      yname: "HF retention (E_hi/E_tot)_sparse / (E_hi/E_tot)_dense, b0=16",
      yunit: "ratio (1 = parity)",
      yscale: "linear, from 0 to 1.22 × max(p75, median, diamond) over all bars",
      encoding: {
        bar: "R^{(c)}_{16}: diffusion estimator, MEDIAN over frames; fill = dataset color alpha 0.9",
        whiskers: "IQR p25–p75 of the per-frame values (numpy percentile, linear interpolation), color #333333, capsize 2.5",
        diamond: "HFR^{(c)}: KPM per-class true-HF share ratio sparse/dense, MEDIAN over frames; black 'D' marker",
        parity_line: "y = 1, black dashed",
        n_label: "text 'n=<n>' at y=0.03, rotated 90°: n = frames where the class is valid (diffusion count)",
        printed_label: "value printed above each bar = max(p75, median, diamond), format {:.3f}, rotated 90° — NOT the median"
      },
      series: [
        {
          ds: "semantickitti",
          display: "SemanticKITTI",
          color: "#4477AA",
          classes: [
            {
              cid: 1,
              name: "Driveable Ground",
              median: 0.4111779508593672,
              p25: 0.33089164211891914,
              p75: 0.5368593408532857,
              diamond: 0.41667119737433955,
              n: 200,
              printed: 0.5368593408532857
            },
            {
              cid: 2,
              name: "Structure",
              median: 0.9804671405577818,
              p25: 0.7774326867095167,
              p75: 1.2488477882415214,
              diamond: 0.8916250392399573,
              n: 200,
              printed: 1.2488477882415214
            },
            {
              cid: 3,
              name: "Vehicle",
              median: 0.690481214028912,
              p25: 0.5372415274785991,
              p75: 0.8556136855519293,
              diamond: 0.6278029970633736,
              n: 198,
              printed: 0.8556136855519293
            },
            {
              cid: 4,
              name: "Nature",
              median: 1.2522642390203698,
              p25: 1.1296510134905897,
              p75: 1.4122360212141876,
              diamond: 1.3314561243336738,
              n: 200,
              printed: 1.4122360212141876
            },
            {
              cid: 5,
              name: "Living Being",
              median: 0.6774198612685095,
              p25: 0.4649433039784949,
              p75: 0.8697878041685877,
              diamond: 0.5772647065342955,
              n: 39,
              printed: 0.8697878041685877
            },
            {
              cid: 6,
              name: "Objects",
              median: 1.0570156755136373,
              p25: 0.8608959296430839,
              p75: 1.322679515125163,
              diamond: 1.0721995012516974,
              n: 131,
              printed: 1.322679515125163
            },
            {
              cid: 7,
              name: "Other Ground",
              median: 0.8771023634131914,
              p25: 0.6616038241716691,
              p75: 1.0832325969812961,
              diamond: 0.9318063036525317,
              n: 200,
              printed: 1.0832325969812961
            }
          ]
        },
        {
          ds: "waymo",
          display: "Waymo",
          color: "#EE6677",
          classes: [
            {
              cid: 1,
              name: "Driveable Ground",
              median: 0.5503380151879192,
              p25: 0.4289333400160326,
              p75: 0.6709231122102027,
              diamond: 0.5610233153902164,
              n: 200,
              printed: 0.6709231122102027
            },
            {
              cid: 2,
              name: "Structure",
              median: 0.9880355622545026,
              p25: 0.8839279760013998,
              p75: 1.0932505474060243,
              diamond: 1.0222714122849546,
              n: 200,
              printed: 1.0932505474060243
            },
            {
              cid: 3,
              name: "Vehicle",
              median: 0.7751551005096604,
              p25: 0.6486592326109178,
              p75: 0.9120889986928399,
              diamond: 0.8045830651878848,
              n: 188,
              printed: 0.9120889986928399
            },
            {
              cid: 4,
              name: "Nature",
              median: 1.1665997554432004,
              p25: 1.0312707685991738,
              p75: 1.3362410935563633,
              diamond: 1.2199856825333113,
              n: 191,
              printed: 1.3362410935563633
            },
            {
              cid: 5,
              name: "Living Being",
              median: 0.7641976320933804,
              p25: 0.6198558883357583,
              p75: 0.8714663948346801,
              diamond: 0.7643133874678274,
              n: 79,
              printed: 0.8714663948346801
            },
            {
              cid: 6,
              name: "Objects",
              median: 0.7826366317422657,
              p25: 0.6265196969550474,
              p75: 0.9431494328890014,
              diamond: 0.7873374988379798,
              n: 190,
              printed: 0.9431494328890014
            },
            {
              cid: 7,
              name: "Other Ground",
              median: 0.9339922828898415,
              p25: 0.7720374650360952,
              p75: 1.136919608078378,
              diamond: 0.9455209417220402,
              n: 200,
              printed: 1.136919608078378
            }
          ]
        },
        {
          ds: "nuscenes",
          display: "nuScenes",
          color: "#228833",
          classes: [
            {
              cid: 1,
              name: "Driveable Ground",
              median: 0.5217859634927181,
              p25: 0.3986877229278004,
              p75: 0.7147778794468141,
              diamond: 0.5110991394109862,
              n: 200,
              printed: 0.7147778794468141
            },
            {
              cid: 2,
              name: "Structure",
              median: 1.8716800186487588,
              p25: 1.627613420695598,
              p75: 2.2075794687799757,
              diamond: 1.9549504156821518,
              n: 200,
              printed: 2.2075794687799757
            },
            {
              cid: 3,
              name: "Vehicle",
              median: 1.2502143527619678,
              p25: 0.8448641861907991,
              p75: 1.6841282450446844,
              diamond: 0.9829703463584243,
              n: 144,
              printed: 1.6841282450446844
            },
            {
              cid: 4,
              name: "Nature",
              median: 2.195202439149051,
              p25: 1.977973875114646,
              p75: 2.429239436622746,
              diamond: 2.58808846761604,
              n: 181,
              printed: 2.58808846761604
            },
            {
              cid: 5,
              name: "Living Being",
              median: 2.2494811446466305,
              p25: 1.3706640792536557,
              p75: 2.8318505646577177,
              diamond: 1.6556434496017338,
              n: 7,
              printed: 2.8318505646577177
            },
            {
              cid: 6,
              name: "Objects",
              median: 2.871448756666535,
              p25: 1.9152163594976745,
              p75: 3.6095060603210096,
              diamond: 2.492089475772102,
              n: 29,
              printed: 3.6095060603210096
            },
            {
              cid: 7,
              name: "Other Ground",
              median: 1.2860106794288546,
              p25: 1.0193275940504396,
              p75: 1.625594427008601,
              diamond: 1.3030600490679074,
              n: 198,
              printed: 1.625594427008601
            }
          ]
        }
      ]
    },
    heatmap: {
      use: "merged_10_columns (this is exactly what the figure draws)",
      why_merged: "bands 1 (λ<0.01) and 2 (0.01–0.03) both lie inside the Jackson kernel-width soft edge (<0.03), so the figure merges them into one '<0.03' column",
      xname: "graph-frequency band λ (bands 1+2 merged)",
      yname: "global signal / class (n = frames where the class exists on both sides)",
      columns: ["<0.03", "0.03–0.08", "0.08–0.15", "0.15–0.25", "0.25–0.4", "0.4–0.6", "0.6–0.85", "0.85–1.15", "1.15–1.5", "1.5–2.02"],
      vmax: 0.3820365948699273,
      cell_text: "G × 1000, format {:.1f}; text white if |G| > 0.55·vmax else #222222; NaN cells show 'N/A' in #666666",
      meaning: "red = dense excess (G>0), blue = sparse excess (G<0)",
      cmap: "RdBu_r (matplotlib)",
      reference_variant: {
        columns: ["0–0.01", "0.01–0.03", "0.03–0.08", "0.08–0.15", "0.15–0.25", "0.25–0.4", "0.4–0.6", "0.6–0.85", "0.85–1.15", "1.15–1.5", "1.5–2.02"],
        vmax: 0.4303054728549028,
        note: "reference only; not drawn. merged column 0 = band 0 + band 1"
      },
      matrices: {
        semantickitti: [
          {
            row: "density",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.3568040277822746, 0.06717437675181755, 0.060589807061731185, 0.05399128721865277, 0.053863266321307945, 0.048338046146292564, 0.03930998721865901, 0.029935921636270093, 0.003612031914723319, -1.0696487179697033e-05]
          },
          {
            row: "normal_z",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.123388540111048, -0.012770808174128106, -0.00743943795635565, -0.002728057907232997, 0.003607447003109158, 0.0144916353266032, 0.03127264170435434, 0.0757316855096691, 0.02190720865199713, -0.000683774046968212]
          },
          {
            row: "surface_variation",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.059403794084478094, -0.00503702654784879, -0.0024179342095600313, 0.0007965862888650107, 0.0033693650020568044, 0.008293848441977487, 0.010721030141753995, 0.03619609609182109, 0.007725638324637125, -0.00024380944922468685]
          },
          {
            row: "Driveable Ground (n=200)",
            kind: "class",
            cid: 1,
            n: 200,
            v: [-0.07378762665510845, 0.011310621074934184, 0.007590749682079769, 0.007231264344548103, 0.007040512325789649, 0.007579791037625885, 0.008022269997693428, 0.021244995119344873, 0.004143272578239202, -0.00037584950514661934]
          },
          {
            row: "Structure (n=200)",
            kind: "class",
            cid: 2,
            n: 200,
            v: [0.039020104823253715, -0.03161946481897544, -0.007390674708017059, -0.00253890349127062, -0.00018281750811914222, 0.0009463695018077403, 0.0011322847643304224, 0.0014578556196622632, -7.582899008167575e-05, -0.0007489251925901476]
          },
          {
            row: "Vehicle (n=198)",
            kind: "class",
            cid: 3,
            n: 198,
            v: [-0.014658178128239358, -0.015054031112619224, -0.003081996452045118, 0.001501183899092461, 0.0035304859573178464, 0.006383916546877008, 0.006372401780180518, 0.010657913768586252, 0.0047549992054439225, -0.0004066954645943094]
          },
          {
            row: "Nature (n=200)",
            kind: "class",
            cid: 4,
            n: 200,
            v: [0.04105435791824945, -0.019857289918123015, -0.006340240490458426, -0.0028333406207798717, -0.0005503005514249723, 0.0005662750059974822, -0.00020355512501621446, -0.009799932948613364, -0.0012826248172298437, -0.0007533484526011609]
          },
          {
            row: "Living Being (n=39)",
            kind: "class",
            cid: 5,
            n: 39,
            v: [-0.03459279445314723, -0.028643835317132254, -0.007866034860069456, 0.0126415886678626, 0.010396798264830469, 0.012260449620598093, 0.011267603835361276, 0.019771534426790905, 0.0051771261577667855, -0.00041243634286114493]
          },
          {
            row: "Objects (n=131)",
            kind: "class",
            cid: 6,
            n: 131,
            v: [0.06013731853773496, -0.047581746424898444, -0.007096533494291325, -0.004424069485021, 0.00029981392503170755, 0.003130662491648564, 0.002929987721216111, -0.003711988721419664, -0.0018390452093114218, -0.0018443993406894075]
          },
          {
            row: "Other Ground (n=200)",
            kind: "class",
            cid: 7,
            n: 200,
            v: [-0.024801558880560667, 0.006771006063436962, 0.0038386385352525343, 0.0024965154765524333, 0.0035224222394942186, 0.004134116080553607, 0.0035201621203639686, 0.002124799891348429, 0.0004558029171943343, -0.002061904443635831]
          }
        ],
        waymo: [
          {
            row: "density",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.33101844620932724, 0.05301897266657758, 0.05235998450098436, 0.051078033456380914, 0.053447964435932506, 0.04877990428027184, 0.04022048084985098, 0.02828716690287631, 0.003798256991752484, 2.7682124700245386e-05]
          },
          {
            row: "normal_z",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.128125436624283, -0.005098451767032431, -0.0018926718259354941, 0.000533661532142375, 0.0054144145484546205, 0.012992187780581126, 0.026027807859316382, 0.07191365149568649, 0.018421142728916196, -0.00018630572784623985]
          },
          {
            row: "surface_variation",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.030826194820780178, -0.003730571380212385, 0.00010467995390351696, 0.0010861097131453753, 0.0030535062863522714, 0.004774462664938247, 0.005444964258065245, 0.017598507878750825, 0.0026392194255270385, -0.00014468397968996432]
          },
          {
            row: "Driveable Ground (n=200)",
            kind: "class",
            cid: 1,
            n: 200,
            v: [-0.05983339630221922, 0.006934681396168124, 0.00487598622102576, 0.004994528747508603, 0.006924723103717865, 0.007730554725440249, 0.007601326590018114, 0.016742377169582698, 0.003825148626961108, 0.00020406972179672837]
          },
          {
            row: "Structure (n=200)",
            kind: "class",
            cid: 2,
            n: 200,
            v: [0.0282094207533765, -0.025299751191268656, -0.004514980106203444, -0.0009400207033053833, 0.0010532878066502464, 0.0021390792728522233, 0.00236932174732318, -0.002835089140869817, -0.00012307095165127444, -5.819748690362343e-05]
          },
          {
            row: "Vehicle (n=188)",
            kind: "class",
            cid: 3,
            n: 188,
            v: [-0.006812491865217974, -0.014844088027284482, -0.007889753716729425, 0.0007328626751709162, 0.005428438910299493, 0.007198678458347038, 0.007330868319981709, 0.004640502290088636, 0.003959651627669959, 0.00025533132767415655]
          },
          {
            row: "Nature (n=191)",
            kind: "class",
            cid: 4,
            n: 191,
            v: [0.030422437443343393, -0.014337015703044163, -0.005071894015952568, -0.003392256325275373, -0.00038509926389028893, 0.0015383071950348878, 0.0010621003416732997, -0.00818161184700353, -0.0015163053656635636, -0.00013866245922208498]
          },
          {
            row: "Living Being (n=79)",
            kind: "class",
            cid: 5,
            n: 79,
            v: [0.012016691558260142, -0.07100825858634446, -0.0028207332475859905, 0.008529913879771436, 0.010007605920198882, 0.012707511834183097, 0.011526518965455965, 0.011885226217986588, 0.006561772071065898, 0.0005937513870084914]
          },
          {
            row: "Objects (n=190)",
            kind: "class",
            cid: 6,
            n: 190,
            v: [-0.014657744012020288, -0.017559065318167238, -0.00024280087818326124, 0.002411922933271082, 0.006856426400574821, 0.008255499653536573, 0.007975592768095306, 0.0036795641267173985, 0.0031397061803244584, 0.00014089814585109843]
          },
          {
            row: "Other Ground (n=200)",
            kind: "class",
            cid: 7,
            n: 200,
            v: [-0.0011836521426220575, -0.004008239515549093, -0.0017343637668116186, -0.0003663651371282998, 0.0010956249357527586, 0.0024838012388268206, 0.002008128394470642, 0.0010100120781448352, 0.0012267391148325698, -0.0005316851999165646]
          }
        ],
        nuscenes: [
          {
            row: "density",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.20736689537955882, 0.019837216358675646, 0.03512976389486943, 0.028794709311301105, 0.03791370299599153, 0.03778168586521804, 0.03541845517417339, 0.028346817466792643, -0.000806870781877568, -0.015048584905585362]
          },
          {
            row: "normal_z",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.15381907457664945, -0.022556048129012515, -0.011388897519676094, -0.0071262047277344046, 0.0015109546495915309, 0.014254302340654515, 0.03427898689616605, 0.11711005338052452, 0.03304957876165626, -0.005313651075520438]
          },
          {
            row: "surface_variation",
            kind: "signal",
            cid: null,
            n: 200,
            v: [-0.019513986741823742, -0.01895872225397397, -0.009005997295868474, -0.002848384636362416, 0.0023142462497409534, 0.0053380967567100215, 0.004604076883857019, 0.033793318315186185, 0.007311606412074865, -0.003034253689540439]
          },
          {
            row: "Driveable Ground (n=200)",
            kind: "class",
            cid: 1,
            n: 200,
            v: [-0.039785608506250784, 0.002946850090268679, 0.003102246342125563, 0.0037264434481016984, 0.004312199302653228, 0.004659353469854181, 0.005111160112999884, 0.016052386938026485, 0.0009266931339918616, -0.0010517243317708013]
          },
          {
            row: "Structure (n=200)",
            kind: "class",
            cid: 2,
            n: 200,
            v: [0.0895884456247106, -0.034255614670559265, -0.0138794547086584, -0.007890876244338067, -0.004357538091380132, -0.0026483720604737783, -0.0016190680161888254, -0.019706402299625926, -0.00249721203696132, -0.0027339074965248896]
          },
          {
            row: "Vehicle (n=144)",
            kind: "class",
            cid: 3,
            n: 144,
            v: [0.11892766637390446, -0.06353428363396614, -0.027608217716279998, -0.013871985057864276, -0.00908359929482947, -0.004732103549615273, -0.002384658373096147, 0.002425637970665587, 0.0011579625174671817, -0.0012964192363859005]
          },
          {
            row: "Nature (n=181)",
            kind: "class",
            cid: 4,
            n: 181,
            v: [0.11143696605327896, -0.0252927621278704, -0.011646573333753477, -0.007484833800627388, -0.0034510398420297835, -0.002369138726757154, -0.004396350315766408, -0.05007674743652307, -0.0037626422306570604, -0.002956878239294212]
          },
          {
            row: "Living Being (n=7)",
            kind: "class",
            cid: 5,
            n: 7,
            v: [0.30672650631810994, -0.13831406275038927, -0.052604161421513716, -0.023018552306465545, -0.024113799013905556, -0.018091672402591717, -0.011714229718769644, -0.032219398786657145, -0.0014172185255231908, -0.005233411392294092]
          },
          {
            row: "Objects (n=29)",
            kind: "class",
            cid: 6,
            n: 29,
            v: [0.3820365948699273, -0.08925111296292443, -0.08701300773185563, -0.06219371519325537, -0.04440259294976926, -0.030871941702608215, -0.021424413371539476, -0.03305064066252293, -0.00473199908739426, -0.009097171208057874]
          },
          {
            row: "Other Ground (n=198)",
            kind: "class",
            cid: 7,
            n: 198,
            v: [0.05003539685733581, -0.011204240519635617, -0.005849695602026485, -0.005469069647920085, -0.0027209154084829835, -0.0010359638836682497, -0.0020498899090596366, -0.013431664378167157, -0.003743159179011179, -0.004530798329364386]
          }
        ]
      }
    },
    b0trend: {
      x: [2, 8, 16, 32],
      xname: "diffusion cutoff b0",
      xscale: "log base 2",
      yname: "median HF retention (sparse/dense)",
      yunit: "ratio (1 = parity, black dashed line)",
      panels: [
        {
          ds: "semantickitti",
          title: "SemanticKITTI",
          color: "#4477AA",
          series: [
            {
              name: "Driveable Ground",
              cid: 1,
              color: "#1f77b4",
              v: [0.4517697323245277, 0.4160969978710506, 0.4111779508593672, 0.4231111704228284],
              n: 200
            },
            {
              name: "Structure",
              cid: 2,
              color: "#ff7f0e",
              v: [0.9422288740397906, 0.9327460041108797, 0.9804671405577818, 1.0513002550945743],
              n: 200
            },
            {
              name: "Vehicle",
              cid: 3,
              color: "#2ca02c",
              v: [0.6397741112424274, 0.6542527498234052, 0.690481214028912, 0.7451770850418279],
              n: 198
            },
            {
              name: "Nature",
              cid: 4,
              color: "#d62728",
              v: [1.3597354245280395, 1.2839175238686278, 1.2522642390203698, 1.275087824867394],
              n: 200
            },
            {
              name: "Living Being",
              cid: 5,
              color: "#9467bd",
              v: [0.6129985120273469, 0.6411881771212091, 0.6774198612685095, 0.7175255188994363],
              n: 39
            },
            {
              name: "Objects",
              cid: 6,
              color: "#8c564b",
              v: [1.075481437752962, 1.0467452244112512, 1.0570156755136373, 1.1054572239927396],
              n: 131
            },
            {
              name: "Other Ground",
              cid: 7,
              color: "#e377c2",
              v: [0.9783204050532919, 0.8980544001647062, 0.8771023634131914, 0.8644439947946605],
              n: 200
            }
          ]
        },
        {
          ds: "waymo",
          title: "Waymo",
          color: "#EE6677",
          series: [
            {
              name: "Driveable Ground",
              cid: 1,
              color: "#1f77b4",
              v: [0.5832407892802018, 0.5602256455448412, 0.5503380151879192, 0.5489647086499299],
              n: 200
            },
            {
              name: "Structure",
              cid: 2,
              color: "#ff7f0e",
              v: [1.0444022608898096, 0.9912379522402208, 0.9880355622545026, 1.001514331740991],
              n: 200
            },
            {
              name: "Vehicle",
              cid: 3,
              color: "#2ca02c",
              v: [0.81920938214438, 0.781103541591604, 0.7751551005096604, 0.7908155953189211],
              n: 188
            },
            {
              name: "Nature",
              cid: 4,
              color: "#d62728",
              v: [1.248954166238829, 1.186390181464779, 1.1665997554432004, 1.1779411312046397],
              n: 191
            },
            {
              name: "Living Being",
              cid: 5,
              color: "#9467bd",
              v: [0.7618424515944703, 0.7688427206395978, 0.7641976320933804, 0.813903962677397],
              n: 79
            },
            {
              name: "Objects",
              cid: 6,
              color: "#8c564b",
              v: [0.8093248157034596, 0.7963675393429137, 0.7826366317422657, 0.7901877028398155],
              n: 190
            },
            {
              name: "Other Ground",
              cid: 7,
              color: "#e377c2",
              v: [0.9609163335149864, 0.9361943148733974, 0.9339922828898415, 0.9314251549317212],
              n: 200
            }
          ]
        },
        {
          ds: "nuscenes",
          title: "nuScenes",
          color: "#228833",
          series: [
            {
              name: "Driveable Ground",
              cid: 1,
              color: "#1f77b4",
              v: [0.5598205044816272, 0.5221802057700845, 0.5217859634927181, 0.5502024985646983],
              n: 200
            },
            {
              name: "Structure",
              cid: 2,
              color: "#ff7f0e",
              v: [2.1691779782516756, 1.9283163391789668, 1.8716800186487588, 1.9266872176591345],
              n: 200
            },
            {
              name: "Vehicle",
              cid: 3,
              color: "#2ca02c",
              v: [1.0614173388906782, 1.1554536935853932, 1.2502143527619678, 1.3880297614140902],
              n: 144
            },
            {
              name: "Nature",
              cid: 4,
              color: "#d62728",
              v: [2.66437083771004, 2.3493613769308035, 2.195202439149051, 2.1216590333657552],
              n: 181
            },
            {
              name: "Living Being",
              cid: 5,
              color: "#9467bd",
              v: [1.9533977932975504, 2.063514084241876, 2.2494811446466305, 2.31770113802363],
              n: 7
            },
            {
              name: "Objects",
              cid: 6,
              color: "#8c564b",
              v: [2.637416874783925, 2.528848943701262, 2.871448756666535, 3.4205140461945853],
              n: 29
            },
            {
              name: "Other Ground",
              cid: 7,
              color: "#e377c2",
              v: [1.3954291997264185, 1.3161778352637776, 1.2860106794288546, 1.2963621493191502],
              n: 198
            }
          ]
        }
      ]
    },
    bands: {
      bands: ["0–0.01", "0.01–0.03", "0.03–0.08", "0.08–0.15", "0.15–0.25", "0.25–0.4", "0.4–0.6", "0.6–0.85", "0.85–1.15", "1.15–1.5", "1.5–2.02"],
      band_edges: [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 0.85, 1.15, 1.5, 2.02],
      xname: "graph-frequency band λ",
      yname: "mean band energy fraction",
      yunit: "fraction of signal energy",
      yscale: "log; values are floored at 1e-8 before plotting",
      marks: {
        dotted_vertical_line: "at x = 1.5 (between band 2 and band 3), color #999999: first two bands lie inside the Jackson kernel-width soft edge < 0.03"
      },
      panels: [
        {
          ds: "semantickitti",
          signal: "density",
          color: "#4477AA",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.4682468686983756, 0.07322869164692393, 0.0874188919765015, 0.07457974317102799, 0.0656169201567681, 0.06531327504572951, 0.059206012126383885, 0.05059024683083711, 0.04922239794623921, 0.006357743442446803, 0.00021920895776623227],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.8762093137211938, 0.022070274406380205, 0.020244515224683943, 0.013989936109296806, 0.011625632938115374, 0.011450008724421533, 0.010867965980091312, 0.011280259612178111, 0.019286476309969124, 0.0027457115277234835, 0.00022990544494592958],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "semantickitti",
          signal: "normal_z",
          color: "#4477AA",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.5019875844064751, 0.03821874805566976, 0.02924425558420071, 0.022694338611499578, 0.024105744232181227, 0.03210664343326859, 0.043778045089376494, 0.06554477365595267, 0.1863960778235546, 0.05561934329277604, 0.00030444581404527935],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.6169466873189142, 0.04664818525427827, 0.042015063758328815, 0.03013377656785521, 0.026833802139414217, 0.02849919643015943, 0.029286409762773285, 0.03427213195159829, 0.11066439231388564, 0.03371213464077893, 0.0009882198610134923],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "semantickitti",
          signal: "surface_variation",
          color: "#4477AA",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.6527637372327836, 0.03576885895589986, 0.03297575694638167, 0.02655565988837635, 0.02598443673042739, 0.02892877527000941, 0.03060761546150806, 0.033768167089120754, 0.10405718459370832, 0.02832071496622669, 0.0002690928645580621],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.7075235793139077, 0.04041281095925408, 0.03801278349423045, 0.02897359409793638, 0.025187850441562375, 0.025559410267952606, 0.02231376701953058, 0.02304713694736677, 0.06786108850188721, 0.020595076641589577, 0.0005129023137827489],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "waymo",
          signal: "density",
          color: "#EE6677",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.49616373795303303, 0.0736262610965629, 0.07545910817162498, 0.06544211000010447, 0.061996542037366875, 0.06396007883073945, 0.05905231446401862, 0.05110682626743301, 0.04684411709581941, 0.00619651172690731, 0.0001523923553902504],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.8673693785139102, 0.033439066745012375, 0.02244013550504734, 0.013082125499120176, 0.010918508580985909, 0.010512114394806983, 0.010272410183746783, 0.010886345417582027, 0.01855695019294309, 0.002398254735154828, 0.00012471023069000504],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "waymo",
          signal: "normal_z",
          color: "#EE6677",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.6122452573431323, 0.041840332794703804, 0.027838903962472617, 0.01921317699352901, 0.018625776834298603, 0.023305967360321044, 0.030941649348057237, 0.04692324375251587, 0.1402120509645037, 0.038654567900589375, 0.00019907274487652573],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.7419230297268825, 0.04028799703523645, 0.03293735572950505, 0.021105848819464507, 0.01809211530215623, 0.017891552811866412, 0.01794946156747612, 0.020895435893199518, 0.06829839946881722, 0.020233425171673192, 0.00038537847272276574],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "waymo",
          signal: "surface_variation",
          color: "#EE6677",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.6816035954238874, 0.03553942078724906, 0.03170495609255999, 0.025056181139952734, 0.024451385974694446, 0.0270118979626789, 0.02839302290240079, 0.03110201259812275, 0.09228772126184792, 0.022695716662126452, 0.0001540891934795253],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.7076515209894629, 0.04031769004245375, 0.03543552747277237, 0.02495150118604921, 0.02336527626154908, 0.023958391676326626, 0.023618560237462533, 0.02565704834005751, 0.07468921338309706, 0.020056497236599434, 0.0002987731731694896],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "nuscenes",
          signal: "density",
          color: "#228833",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.614165458555025, 0.05364249531759477, 0.06143986435704434, 0.04984906735351489, 0.046982778250440525, 0.04723524206471574, 0.04275995394609953, 0.0376497730354462, 0.04080956272181658, 0.00524579949179248, 0.0002200049055094175],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.856973482783724, 0.018201366468455143, 0.04160264799836872, 0.014719303458645423, 0.01818806893913943, 0.009321539068724228, 0.0049782680808814666, 0.0022313178612728226, 0.012462745255023944, 0.00605267027367005, 0.01526858981109478],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "nuscenes",
          signal: "normal_z",
          color: "#228833",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.5968892175209989, 0.022735986601497366, 0.019362979654463926, 0.015225039448623674, 0.017424255093048523, 0.024760992667363993, 0.03516469863155296, 0.05414271805561025, 0.1613305677624049, 0.05245365703845197, 0.0005098875249835935],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.7257013167473112, 0.04774296195183392, 0.04191902778347645, 0.026613936968299765, 0.024550459820782945, 0.02325003801777246, 0.02091039629089845, 0.019863731159444144, 0.04422051438188043, 0.01940407827679569, 0.0058235386005040315],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        },
        {
          ds: "nuscenes",
          signal: "surface_variation",
          color: "#228833",
          series: [
            {
              name: "Dense",
              side: "dense",
              v: [0.6876419240716801, 0.027621060474428572, 0.02907040202778374, 0.025103184717564966, 0.02544211657081312, 0.027971610728959494, 0.02826587547201437, 0.02938041109153002, 0.09379290758597898, 0.025387544888795986, 0.00032296236945058616],
              style: "dataset color, alpha 0.92, edge #555555"
            },
            {
              name: "Sparse",
              side: "sparse",
              v: [0.6777533339851519, 0.057023637302780456, 0.04802912428175768, 0.03410918201343344, 0.028290501207175557, 0.025657364479218563, 0.022927778715304346, 0.024776334207672973, 0.0599995892707928, 0.01807593847672111, 0.0033572160589910205],
              style: "dataset color, alpha 0.38, hatch '///'"
            }
          ]
        }
      ]
    }
  };

  /* ----------------------------------------------------------------- style */

  var STYLE_ID = 'lqc-style';

  var CSS = [
    '.lqc{',
    '  position:relative;margin:0;padding:0;',
    '  --lqc-sans:var(--sans,\'Inter\',system-ui,sans-serif);',
    '  --lqc-mono:var(--mono,\'JetBrains Mono\',ui-monospace,monospace);',
    '  --lqc-ink:var(--text,#17202c);',
    '  --lqc-muted:var(--muted,#647184);',
    '  --lqc-hair:var(--line,rgba(52,72,96,.12));',
    '  --lqc-grid:rgba(52,72,96,.13);',
    '  --lqc-axis:rgba(52,72,96,.34);',
    '  --lqc-edge:#555555;',
    '  --lqc-whisk:#333333;',
    '  --lqc-mark:#111111;',
    '  --lqc-halo:#ffffff;',
    '  --lqc-eyebrow:#377fba;',
    '  --lqc-link:#2f78bf;',
    '  --lqc-hover:#58b9eb;',
    '  color:var(--lqc-ink);font-family:var(--lqc-sans);',
    '}',
    '.lqc[data-lqc-theme="dark"]{',
    '  --lqc-ink:#dbe7f5;--lqc-muted:#9cc8f3;',
    '  --lqc-hair:rgba(190,215,245,.18);--lqc-grid:rgba(190,215,245,.15);',
    '  --lqc-axis:rgba(190,215,245,.40);--lqc-edge:#8ea8c6;--lqc-whisk:#c6d8ee;',
    '  --lqc-mark:#eaf2ff;--lqc-halo:#0c1322;--lqc-eyebrow:#9cc8f3;',
    '  --lqc-link:#56adf0;--lqc-hover:#bfe0ff;',
    '}',
    '.lqc-fig{margin:0;padding:0}',
    '.lqc-panels{display:grid;gap:14px;align-items:start;',
    '  grid-template-columns:repeat(var(--lqc-cols,1),minmax(0,1fr))}',
    '.lqc-panels>svg{display:block;width:100%;height:auto;overflow:visible;',
    '  max-width:var(--lqc-pmax,none);margin:0}',
    '.lqc-anim .lqc-panels>svg{opacity:0;transform:translateY(5px);',
    '  transition:opacity .45s ease,transform .45s ease}',
    '.lqc-anim.lqc-in .lqc-panels>svg{opacity:1;transform:none}',
    '.lqc text{font-family:var(--lqc-mono)}',
    '.lqc .t-tick{font-size:9.5px;fill:var(--lqc-muted)}',
    '.lqc .t-pan{font-size:10.5px;font-weight:600;fill:var(--lqc-ink)}',
    '.lqc .t-cat{font-size:9.5px;fill:var(--lqc-ink)}',
    '.lqc .t-sub{font-size:8px;fill:var(--lqc-muted)}',
    '.lqc .t-dim{font-size:7.6px;fill:var(--lqc-muted)}',
    '.lqc .t-val{font-size:8.6px;fill:var(--lqc-ink)}',
    '.lqc .t-par{font-size:7.6px;fill:var(--lqc-muted);stroke:var(--lqc-halo);',
    '  stroke-width:2.4px;paint-order:stroke;stroke-linejoin:round}',
    '.lqc .t-n{font-size:8px;fill:var(--lqc-ink);stroke:var(--lqc-halo);',
    '  stroke-width:2.2px;paint-order:stroke;stroke-linejoin:round}',
    '.lqc .t-cell{font-weight:600}',
    '.lqc .ax{fill:none;stroke:var(--lqc-axis);stroke-width:1}',
    '.lqc .gr{fill:none;stroke:var(--lqc-grid);stroke-width:1;shape-rendering:crispEdges}',
    '.lqc .par{fill:none;stroke:var(--lqc-mark);stroke-width:1;stroke-dasharray:4 3;opacity:.65}',
    '.lqc .soft{fill:none;stroke:#999999;stroke-width:1;stroke-dasharray:2 2}',
    '.lqc .dia{fill:var(--lqc-mark);stroke:var(--lqc-halo);stroke-width:.7}',
    '.lqc .wk{fill:none;stroke:var(--lqc-whisk);stroke-width:1.1}',
    '.lqc .edge{stroke:var(--lqc-edge);stroke-width:.8}',
    '.lqc .cellsep{stroke:var(--lqc-halo);stroke-width:.6;stroke-opacity:.55}',
    '.lqc .hit{fill:#ffffff;fill-opacity:0;stroke:none;pointer-events:all}',
    '.lqc-legend{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;',
    '  margin:0 0 9px;font-family:var(--lqc-mono);font-size:11px;color:var(--lqc-muted)}',
    '.lqc-key{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}',
    '.lqc-key svg{display:block;flex:0 0 auto}',
    '.lqc-axes{font-family:var(--lqc-mono);font-size:10.5px;line-height:1.5;',
    '  color:var(--lqc-muted);margin:0 0 9px}',
    '.lqc-note{font-family:var(--lqc-sans);font-size:12.5px;line-height:1.55;',
    '  color:var(--lqc-muted);margin:12px 0 0;max-width:80ch}',
    '.lqc-note b{font-family:var(--lqc-mono);font-size:10px;font-weight:600;',
    '  letter-spacing:.09em;text-transform:uppercase;color:var(--lqc-eyebrow);',
    '  margin-right:.45em}',
    '.lqc-foot{margin:10px 0 0}',
    '.lqc-toggle{font-family:var(--lqc-mono);font-size:10.5px;letter-spacing:.06em;',
    '  text-transform:uppercase;color:var(--lqc-eyebrow);background:transparent;',
    '  border:1px solid var(--lqc-hair);border-radius:999px;padding:5px 11px;cursor:pointer}',
    '.lqc-toggle:hover{color:var(--lqc-hover);border-color:var(--lqc-hover)}',
    '.lqc-toggle:focus-visible{outline:2px solid var(--lqc-link);outline-offset:2px}',
    '.lqc-sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;',
    '  clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}',
    '.lqc-sr.is-open{position:static;width:auto;height:auto;margin:14px 0 0;',
    '  overflow:auto;clip:auto;clip-path:none;white-space:normal}',
    '.lqc table{border-collapse:collapse;font-family:var(--lqc-mono);font-size:11px;width:100%}',
    '.lqc caption{text-align:left;font-size:11px;color:var(--lqc-muted);padding:0 0 6px}',
    '.lqc th,.lqc td{border-bottom:1px solid var(--lqc-hair);padding:3px 8px 3px 0;',
    '  text-align:right;white-space:nowrap}',
    '.lqc th:first-child,.lqc td:first-child{text-align:left}',
    '.lqc thead th{color:var(--lqc-muted);font-weight:600}',
    '.lqc-tip{position:absolute;z-index:6;left:0;top:0;pointer-events:none;opacity:0;',
    '  transform:translate(-50%,-112%);background:#0c1322;color:#dbe7f5;',
    '  font-family:var(--lqc-mono);font-size:10.5px;line-height:1.5;padding:6px 9px;',
    '  border-radius:8px;box-shadow:0 10px 24px -14px rgba(12,19,34,.8);',
    '  white-space:pre;max-width:17rem;transition:opacity .12s ease}',
    '.lqc-tip.is-on{opacity:1}',
    '@media (prefers-reduced-motion:reduce){',
    '  .lqc-anim .lqc-panels>svg{opacity:1 !important;transform:none !important}',
    '  .lqc *{animation:none !important;transition:none !important}',
    '}'
  ].join('\n');

  /* --------------------------------------------------------------- dom util */

  var SVGNS = 'http://www.w3.org/2000/svg';
  var UID = 0;

  function doc() {
    return global.document || null;
  }

  function warn(where, err) {
    if (global.console && global.console.warn) {
      global.console.warn('[LiEQCharts] ' + where + ': ' + (err && err.message ? err.message : err));
    }
  }

  function uid(prefix) {
    UID += 1;
    return 'lqc-' + prefix + '-' + UID;
  }

  function injectCSS() {
    var D = doc();
    if (!D || !D.createElement) return;
    if (D.getElementById && D.getElementById(STYLE_ID)) return;
    var s = D.createElement('style');
    s.setAttribute('id', STYLE_ID);
    s.appendChild(D.createTextNode(CSS));
    (D.head || D.documentElement || D.body).appendChild(s);
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function hEl(tag, cls, parent, content) {
    var n = doc().createElement(tag);
    if (cls) n.setAttribute('class', cls);
    if (content !== undefined && content !== null && content !== '') {
      n.appendChild(doc().createTextNode(String(content)));
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function sEl(tag, attrs, parent) {
    var n = doc().createElementNS(SVGNS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (attrs.hasOwnProperty(k) && attrs[k] !== null && attrs[k] !== undefined) {
          n.setAttribute(k, attrs[k]);
        }
      }
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function sTxt(parent, x, y, content, cls, attrs) {
    var a = attrs || {};
    a.x = r2(x);
    a.y = r2(y);
    if (cls) a['class'] = cls;
    var t = sEl('text', a, parent);
    t.appendChild(doc().createTextNode(String(content)));
    return t;
  }

  function sSpan(parent, content, cls) {
    var t = sEl('tspan', cls ? { 'class': cls } : null, parent);
    t.appendChild(doc().createTextNode(String(content)));
    return t;
  }

  function sLine(parent, x1, y1, x2, y2, cls, attrs) {
    var a = attrs || {};
    a.x1 = r2(x1); a.y1 = r2(y1); a.x2 = r2(x2); a.y2 = r2(y2);
    if (cls) a['class'] = cls;
    return sEl('line', a, parent);
  }

  function sRect(parent, x, y, w, h, attrs) {
    var a = attrs || {};
    a.x = r2(x); a.y = r2(y); a.width = r2(Math.max(0, w)); a.height = r2(Math.max(0, h));
    return sEl('rect', a, parent);
  }

  function setStyle(node, str) {
    if (node && node.setAttribute) node.setAttribute('style', str);
  }

  function setCls(node, str) {
    if (node && node.setAttribute) node.setAttribute('class', str);
  }

  /* --------------------------------------------------------------- numbers */

  function r2(v) {
    return Math.round(v * 100) / 100;
  }

  function trimZeros(s) {
    if (s.indexOf('.') < 0) return s;
    s = s.replace(/0+$/, '');
    if (s.charAt(s.length - 1) === '.') s = s.slice(0, -1);
    return s;
  }

  /* Python's "{:.<p>g}" format, so bar labels match the published figure. */
  function gf(v, p) {
    if (v === null || v === undefined || !isFinite(v)) return 'n/a';
    if (v === 0) return '0';
    p = p || 3;
    var es = Math.abs(v).toExponential(p - 1);
    var parts = es.split('e');
    var e = parseInt(parts[1], 10);
    var sign = v < 0 ? '-' : '';
    if (e < -4 || e >= p) {
      var mant = trimZeros(parts[0]);
      var ae = Math.abs(e);
      return sign + mant + 'e' + (e < 0 ? '-' : '+') + (ae < 10 ? '0' + ae : String(ae));
    }
    var dec = p - 1 - e;
    if (dec < 0) dec = 0;
    return sign + trimZeros(Math.abs(v).toFixed(dec));
  }

  function fx(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return 'n/a';
    return Number(v).toFixed(d);
  }

  function sgn(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return 'n/a';
    return (v > 0 ? '+' : '') + Number(v).toFixed(d);
  }

  function minOf(arr) {
    var m = Infinity, i;
    for (i = 0; i < arr.length; i++) if (arr[i] !== null && isFinite(arr[i]) && arr[i] < m) m = arr[i];
    return m;
  }

  function maxOf(arr) {
    var m = -Infinity, i;
    for (i = 0; i < arr.length; i++) if (arr[i] !== null && isFinite(arr[i]) && arr[i] > m) m = arr[i];
    return m;
  }

  function log10(v) {
    return Math.log(v) / Math.LN10;
  }

  /* ---------------------------------------------------------------- scales */

  function linScale(d0, d1, r0, r1) {
    var m = (r1 - r0) / ((d1 - d0) || 1);
    return function (v) { return r0 + (v - d0) * m; };
  }

  function logScale(d0, d1, r0, r1) {
    var l0 = Math.log(d0), m = (r1 - r0) / ((Math.log(d1) - l0) || 1);
    return function (v) {
      var x = (v > 0) ? v : d0;
      return r0 + (Math.log(x) - l0) * m;
    };
  }

  function niceStep(span, target) {
    var raw = Math.abs(span) / Math.max(1, target || 5);
    if (!(raw > 0)) return 1;
    var mag = Math.pow(10, Math.floor(log10(raw)));
    var n = raw / mag;
    var s = n < 1.5 ? 1 : (n < 3 ? 2 : (n < 7 ? 5 : 10));
    return s * mag;
  }

  function linTicks(lo, hi, target) {
    var st = niceStep(hi - lo, target), out = [], k = Math.ceil(lo / st - 1e-9), v, i;
    for (i = 0; i < 200; i++) {
      v = (k + i) * st;
      if (v > hi + st * 1e-9) break;
      out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    }
    return { ticks: out, step: st };
  }

  function tickDec(step) {
    var d = Math.ceil(-log10(step) - 1e-9);
    return d > 0 ? d : 0;
  }

  function logTicks(lo, hi) {
    var e0 = Math.floor(log10(lo) + 1e-9), e1 = Math.ceil(log10(hi) - 1e-9), e, i, v;
    var majors = [], all = [], subs = [1, 2, 5];
    for (e = e0; e <= e1; e++) {
      v = Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) majors.push(v);
    }
    if (majors.length >= 4) return majors;
    for (e = e0; e <= e1; e++) {
      for (i = 0; i < subs.length; i++) {
        v = subs[i] * Math.pow(10, e);
        if (v >= lo * 0.999 && v <= hi * 1.001) all.push(v);
      }
    }
    return all.length ? all : majors;
  }

  /* domain padded by padFrac; snap=true rounds the ends onto the tick step. */
  function niceDomain(min, max, padFrac, target, snap) {
    var span = (max - min) || Math.abs(max) || 1;
    var lo = min - span * padFrac, hi = max + span * padFrac;
    if (!snap) return [lo, hi];
    var st = niceStep(hi - lo, target);
    return [Math.floor(lo / st) * st, Math.ceil(hi / st) * st];
  }

  function decadeDomain(min, max, headroom) {
    var lo = Math.pow(10, Math.floor(log10(min) + 1e-9));
    var hi = max * (headroom || 1);
    var hiDec = Math.pow(10, Math.ceil(log10(hi) - 1e-9));
    return [lo, headroom ? hi : hiDec];
  }

  /* --------------------------------------------------------- colour helpers */

  /* matplotlib RdBu_r anchors (ColorBrewer RdBu 11, reversed): blue -> red. */
  var RDBU_R = [
    [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
    [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31]
  ];

  function rdbuR(t) {
    var n = RDBU_R.length - 1;
    var x = t < 0 ? 0 : (t > 1 ? 1 : t);
    var p = x * n, i = Math.floor(p), f = p - i, a, b;
    if (i >= n) { i = n - 1; f = 1; }
    a = RDBU_R[i]; b = RDBU_R[i + 1];
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' +
      Math.round(a[1] + (b[1] - a[1]) * f) + ',' +
      Math.round(a[2] + (b[2] - a[2]) * f) + ')';
  }

  function hatchDef(defs, color, id) {
    var p = sEl('pattern', {
      id: id, width: 5, height: 5,
      patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'
    }, defs);
    sEl('line', {
      x1: 0, y1: 0, x2: 0, y2: 5, stroke: color,
      'stroke-width': 1.5, 'stroke-opacity': 0.72
    }, p);
    return 'url(#' + id + ')';
  }

  /* dense = solid fill alpha .92; sparse = alpha .38 + '///' hatch. Never colour. */
  function fillBar(parent, x, y, w, h, color, side, hatchUrl, alpha) {
    var g = sEl('g', null, parent);
    var op = alpha !== undefined ? alpha : (side === 'dense' ? 0.92 : 0.38);
    sRect(g, x, y, w, h, { fill: color, 'fill-opacity': op });
    if (side !== 'dense' && hatchUrl) sRect(g, x, y, w, h, { fill: hatchUrl });
    sRect(g, x, y, w, h, { fill: 'none', 'class': 'edge' });
    return g;
  }

  /* -------------------------------------------------------------- a11y bits */

  function prefersReduced() {
    try {
      return !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) {
      return false;
    }
  }

  function svgRoot(parent, w, h, title, desc) {
    var tid = uid('t'), did = uid('d');
    var svg = sEl('svg', {
      viewBox: '0 0 ' + r2(w) + ' ' + r2(h),
      role: 'img',
      'aria-labelledby': tid + ' ' + did,
      preserveAspectRatio: 'xMidYMid meet',
      focusable: 'false'
    }, parent);
    var t = sEl('title', { id: tid }, svg);
    t.appendChild(doc().createTextNode(String(title)));
    var d = sEl('desc', { id: did }, svg);
    d.appendChild(doc().createTextNode(String(desc)));
    return svg;
  }

  function buildTable(host, caption, head, rows) {
    var tbl = hEl('table', null, host);
    hEl('caption', null, tbl, caption);
    var thead = hEl('thead', null, tbl);
    var tr = hEl('tr', null, thead), i, j, r;
    for (i = 0; i < head.length; i++) {
      hEl('th', null, tr, head[i]).setAttribute('scope', 'col');
    }
    var tb = hEl('tbody', null, tbl);
    for (i = 0; i < rows.length; i++) {
      r = hEl('tr', null, tb);
      for (j = 0; j < rows[i].length; j++) {
        if (j === 0) {
          hEl('th', null, r, rows[i][j]).setAttribute('scope', 'row');
        } else {
          hEl('td', null, r, rows[i][j]);
        }
      }
    }
    return tbl;
  }

  /* --------------------------------------------------------- figure notes */

  var EXTRA = {
    qprofiles: 'The x axis is the diffusion step t, not the diffusion cutoff b0.',
    rolloff: 'Both panels use a logarithmic y axis; bar labels follow the published 3-significant-digit format.',
    retention: 'The number printed above each bar is max(p75, median, HFR^(c)), exactly as in the published figure - it is not the median. The median, the IQR and the full-spectrum estimate are in the tooltip and in the data table.',
    heatmap: 'The first two bands (lambda < 0.01 and 0.01-0.03) are merged into one "<0.03" column because both fall inside the Jackson kernel soft edge; this merged 10-column variant is what the published figure draws. Cell numbers are G x 1000; they are rounded to whole numbers when a cell is too narrow for one decimal, and the exact values are in the data table.',
    b0trend: 'Lines are coloured by class (matplotlib tab10); the y axis is shared across the three datasets and the dashed line marks parity.',
    bands: 'The dotted vertical line sits at the edge of the Jackson kernel soft edge (lambda = 0.03): the first two bands lie inside it. The y axis is logarithmic, shared per dataset row.'
  };

  function noteText(key) {
    var m = DATA.meta, b = DATA.basis[key] || {};
    var out = [];
    out.push(m.frames_per_dataset + ' frames per dataset (' + m.frame_selection + ').');
    out.push('Dense = ' + m.dense + '; sparse = ' + m.sparse + '.');
    out.push('Statistic: ' + b.statistic + '.');
    out.push('Spectrum: ' + m.spectrum + '.');
    out.push('Labels: ' + m.label_mapping + '; ' + m.class_convention + '.');
    if (b.note) out.push('Caveat: ' + b.note + '.');
    if (EXTRA[key]) out.push(EXTRA[key]);
    return out.join(' ');
  }

  /* ------------------------------------------------------------ legend bits */

  function keyBox(legend, label) {
    var k = hEl('span', 'lqc-key', legend);
    var svg = sEl('svg', { width: 22, height: 12, viewBox: '0 0 22 12', 'aria-hidden': 'true' }, k);
    k.appendChild(doc().createTextNode(label));
    return svg;
  }

  function legendFillKeys(legend, pairs) {
    var order = DATA.meta.dataset_order, i, ds, info, svg;
    for (i = 0; i < order.length; i++) {
      ds = order[i].toLowerCase();
      info = DATA.datasets[ds];
      svg = keyBox(legend, info.display);
      if (pairs === false) {
        sEl('rect', { x: 0, y: 1, width: 21, height: 10, fill: info.color, 'fill-opacity': 0.9 }, svg);
      } else {
        sEl('rect', { x: 0, y: 1, width: 10, height: 10, fill: info.color, 'fill-opacity': 0.92 }, svg);
        sEl('rect', { x: 11, y: 1, width: 10, height: 10, fill: info.color, 'fill-opacity': 0.38 }, svg);
        sEl('path', {
          d: 'M11 11 L16 1 M16 11 L21 1', stroke: info.color,
          'stroke-width': 1.2, 'stroke-opacity': 0.75, fill: 'none'
        }, svg);
      }
    }
    if (pairs === false) return;
    var s1 = keyBox(legend, 'dense (solid)');
    sEl('rect', { x: 0, y: 1, width: 20, height: 10, fill: 'currentColor', 'fill-opacity': 0.92 }, s1);
    var s2 = keyBox(legend, 'sparse (hatched)');
    sEl('rect', { x: 0, y: 1, width: 20, height: 10, fill: 'currentColor', 'fill-opacity': 0.3 }, s2);
    sEl('path', {
      d: 'M0 11 L11 0 M9 11 L20 0', stroke: 'currentColor',
      'stroke-width': 1.2, 'stroke-opacity': 0.8, fill: 'none'
    }, s2);
  }

  function legendLineKeys(legend) {
    var order = DATA.meta.dataset_order, i, ds, info, svg;
    for (i = 0; i < order.length; i++) {
      ds = order[i].toLowerCase();
      info = DATA.datasets[ds];
      svg = keyBox(legend, info.display);
      sEl('line', { x1: 0, y1: 6, x2: 21, y2: 6, stroke: info.color, 'stroke-width': 2 }, svg);
      sEl('circle', { cx: 10.5, cy: 6, r: 2.6, fill: info.color }, svg);
    }
    var a = keyBox(legend, 'dense (solid, circles)');
    sEl('line', { x1: 0, y1: 6, x2: 21, y2: 6, stroke: 'currentColor', 'stroke-width': 1.8 }, a);
    sEl('circle', { cx: 10.5, cy: 6, r: 2.6, fill: 'currentColor' }, a);
    var b = keyBox(legend, 'sparse (dashed, squares)');
    sEl('line', {
      x1: 0, y1: 6, x2: 21, y2: 6, stroke: 'currentColor',
      'stroke-width': 1.8, 'stroke-dasharray': '5 3'
    }, b);
    sEl('rect', { x: 8, y: 3.5, width: 5, height: 5, fill: 'currentColor' }, b);
  }

  function legendClassKeys(legend) {
    var names = DATA.retention.classes, i, svg, color;
    for (i = 0; i < names.length; i++) {
      color = DATA.classColors[names[i]];
      svg = keyBox(legend, names[i]);
      sEl('line', { x1: 0, y1: 6, x2: 21, y2: 6, stroke: color, 'stroke-width': 2 }, svg);
      sEl('circle', { cx: 10.5, cy: 6, r: 2.6, fill: color }, svg);
    }
  }

  function legendColorbar(legend, vmax) {
    var k = hEl('span', 'lqc-key', legend);
    var gid = uid('cb');
    var svg = sEl('svg', { width: 190, height: 26, viewBox: '0 0 190 26', 'aria-hidden': 'true' }, k);
    var defs = sEl('defs', null, svg);
    var grad = sEl('linearGradient', { id: gid, x1: '0%', y1: '0%', x2: '100%', y2: '0%' }, defs);
    var i, n = 24;
    for (i = 0; i <= n; i++) {
      sEl('stop', {
        offset: (i / n * 100).toFixed(1) + '%',
        'stop-color': rdbuR(i / n)
      }, grad);
    }
    sEl('rect', {
      x: 0, y: 1, width: 190, height: 9, fill: 'url(#' + gid + ')',
      stroke: '#555555', 'stroke-opacity': 0.35
    }, svg);
    sTxt(svg, 0, 22, gf(-vmax * 1000, 3), 't-dim', { 'text-anchor': 'start' });
    sTxt(svg, 95, 22, '0', 't-dim', { 'text-anchor': 'middle' });
    sTxt(svg, 190, 22, '+' + gf(vmax * 1000, 3), 't-dim', { 'text-anchor': 'end' });
    k.appendChild(doc().createTextNode('G x 1000 (blue = sparse excess, red = dense excess)'));
    return svg;
  }

  /* -------------------------------------------------------------- tooltips */

  function tipFor(node, text) {
    node.__lqcTip = text;
    return node;
  }

  function findTip(node, root) {
    var n = node, i = 0;
    while (n && n !== root && i < 8) {
      if (n.__lqcTip) return n;
      n = n.parentNode;
      i += 1;
    }
    return null;
  }

  /* One shared scroll handler: re-rendering must not pile up window listeners. */
  var ACTIVE_TIP = null;
  var SCROLL_BOUND = false;

  function hideActiveTip() {
    if (ACTIVE_TIP) {
      setCls(ACTIVE_TIP, 'lqc-tip');
      ACTIVE_TIP = null;
    }
  }

  function bindScrollOnce() {
    if (SCROLL_BOUND || !global.addEventListener) return;
    SCROLL_BOUND = true;
    try {
      global.addEventListener('scroll', hideActiveTip, { passive: true });
    } catch (e) { /* ignore */ }
  }

  function attachTips(fig, panels, tip) {
    if (!panels || !panels.addEventListener) return;
    var showing = null;

    function place(ev) {
      if (!fig.getBoundingClientRect) return;
      var box = fig.getBoundingClientRect();
      var cx = (ev.clientX !== undefined ? ev.clientX : 0) - box.left;
      var cy = (ev.clientY !== undefined ? ev.clientY : 0) - box.top;
      setStyle(tip, 'left:' + Math.round(cx) + 'px;top:' + Math.round(cy - 6) + 'px');
    }

    function show(target, ev) {
      if (showing === target) { place(ev); return; }
      showing = target;
      clear(tip);
      var lines = String(target.__lqcTip).split('\n'), i;
      for (i = 0; i < lines.length; i++) {
        if (i) hEl('br', null, tip);
        tip.appendChild(doc().createTextNode(lines[i]));
      }
      setCls(tip, 'lqc-tip is-on');
      ACTIVE_TIP = tip;
      place(ev);
    }

    function hide() {
      showing = null;
      setCls(tip, 'lqc-tip');
      if (ACTIVE_TIP === tip) ACTIVE_TIP = null;
    }

    function onMove(ev) {
      var t = findTip(ev.target, panels);
      if (t) show(t, ev); else hide();
    }

    if (global.PointerEvent) {
      panels.addEventListener('pointermove', onMove, { passive: true });
      panels.addEventListener('pointerdown', onMove, { passive: true });
      panels.addEventListener('pointerleave', hide, { passive: true });
      panels.addEventListener('pointercancel', hide, { passive: true });
    } else {
      panels.addEventListener('mousemove', onMove);
      panels.addEventListener('mouseleave', hide);
      panels.addEventListener('touchstart', function (ev) {
        var t0 = ev.touches && ev.touches[0];
        if (!t0) return;
        var t = findTip(ev.target, panels);
        if (t) show(t, { clientX: t0.clientX, clientY: t0.clientY, target: ev.target });
      }, { passive: true });
      panels.addEventListener('touchend', hide, { passive: true });
    }
    bindScrollOnce();
  }

  /* ------------------------------------------------------------- scaffold */

  function scaffold(mount, cfg) {
    var D = doc();
    var fig = D.createElement('div');
    setCls(fig, 'lqc' + (prefersReduced() ? '' : ' lqc-anim'));
    fig.setAttribute('data-chart', cfg.key);
    if (cfg.theme === 'dark') fig.setAttribute('data-lqc-theme', 'dark');

    var legend = hEl('div', 'lqc-legend', fig);
    var axes = hEl('p', 'lqc-axes', fig, cfg.axes);

    /* <figcaption> must be the first or last child of its <figure>. */
    var inner = hEl('figure', 'lqc-fig', fig);
    var panels = hEl('div', 'lqc-panels', inner);
    setStyle(panels, '--lqc-cols:' + cfg.cols + (cfg.panelMax ? ';--lqc-pmax:' + cfg.panelMax + 'px' : ''));

    var note = hEl('figcaption', 'lqc-note', inner);
    hEl('b', null, note, cfg.noteLabel || 'Measurement basis');
    note.appendChild(D.createTextNode(noteText(cfg.key)));

    var foot = hEl('div', 'lqc-foot', fig);
    var btn = hEl('button', 'lqc-toggle', foot, 'Show data table');
    btn.setAttribute('type', 'button');
    var tableId = uid('tbl');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-controls', tableId);

    var srBox = hEl('div', 'lqc-sr', fig);
    srBox.setAttribute('id', tableId);

    var tip = hEl('div', 'lqc-tip', fig);
    tip.setAttribute('aria-hidden', 'true');

    if (btn.addEventListener) {
      btn.addEventListener('click', function () {
        var open = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', open ? 'false' : 'true');
        setCls(srBox, open ? 'lqc-sr' : 'lqc-sr is-open');
        clear(btn);
        btn.appendChild(D.createTextNode(open ? 'Show data table' : 'Hide data table'));
      });
    }

    mount.appendChild(fig);
    attachTips(fig, panels, tip);

    if (!prefersReduced() && global.requestAnimationFrame) {
      global.requestAnimationFrame(function () {
        global.requestAnimationFrame(function () { setCls(fig, 'lqc lqc-anim lqc-in'); });
      });
    } else {
      setCls(fig, 'lqc lqc-in');
    }

    return { fig: fig, legend: legend, axes: axes, panels: panels, note: note, table: srBox, tip: tip };
  }

  /* ------------------------------------------------------- shared plumbing */

  function resolve(el) {
    if (!el) return null;
    if (typeof el === 'string') {
      var D = doc();
      if (!D || !D.getElementById) return null;
      el = D.getElementById(el.charAt(0) === '#' ? el.slice(1) : el);
    }
    return (el && el.appendChild) ? el : null;
  }

  function measure(mount, opts) {
    var w = (opts && opts.width) || 0;
    if (!w && mount) {
      w = mount.clientWidth || 0;
      if (!w && mount.getBoundingClientRect) w = Math.round(mount.getBoundingClientRect().width || 0);
    }
    if (!w || w < 140) w = (opts && opts.fallbackWidth) || 900;
    return Math.round(Math.min(w, 1180));
  }

  function themeOf(mount, opts) {
    if (opts && opts.theme) return opts.theme;
    var n = mount, i = 0;
    while (n && n.getAttribute && i < 8) {
      var t = n.getAttribute('data-lqc-theme') || n.getAttribute('data-lq-theme') ||
        n.getAttribute('data-theme');
      if (t === 'dark' || t === 'light') return t;
      n = n.parentNode;
      i += 1;
    }
    return 'light';
  }

  function gridCols(w, three, two) {
    if (w >= three) return 3;
    if (w >= two) return 2;
    return 1;
  }

  function panelWidth(w, cols, gap, min, max) {
    var pw = Math.floor((w - gap * (cols - 1)) / cols);
    if (pw < min) pw = min;
    if (max && pw > max) pw = max;
    return pw;
  }

  function watch(mount, paint) {
    if (mount.__lqcWatching) return;
    var timer = null;
    function kick() {
      if (timer) global.clearTimeout(timer);
      timer = global.setTimeout(function () {
        timer = null;
        try { paint(); } catch (e) { warn('resize', e); }
      }, 180);
    }
    if (global.ResizeObserver) {
      try {
        var ro = new global.ResizeObserver(kick);
        ro.observe(mount);
        mount.__lqcWatching = ro;
        return;
      } catch (e) { /* fall through */ }
    }
    if (global.addEventListener) {
      global.addEventListener('resize', kick);
      mount.__lqcWatching = true;
    }
  }

  function run(el, key, opts, draw) {
    var mount = resolve(el);
    if (!mount) return null;
    try {
      injectCSS();
      var o = opts || {};
      var paint = function () {
        var w = measure(mount, o);
        var theme = themeOf(mount, o);
        var sig = key + '|' + theme + '|' + Math.round(w / 40);
        if (mount.__lqcSig === sig && !o.force) return mount.__lqcFig;
        clear(mount);
        var fig = draw(mount, w, theme, o);
        mount.__lqcSig = sig;
        mount.__lqcFig = fig;
        return fig;
      };
      var out = paint();
      if (o.watch !== false) watch(mount, paint);
      return out;
    } catch (err) {
      warn(key, err);
      return null;
    }
  }

  /* =================================================== 1. q profiles ===== */

  function drawQProfiles(mount, W, theme) {
    var F = DATA.qprofiles;
    var cols = gridCols(W, 980, 640);
    var pw = panelWidth(W, cols, 14, 264, 420);
    var ph = 238;
    var pad = { l: 42, r: 12, t: 26, b: 30 };

    var ctx = scaffold(mount, {
      key: 'qprofiles', cols: cols, theme: theme, panelMax: pw,
      axes: 'x: ' + F.xname + ' (t = 1, 4, 16, 32) · y: ' + F.yname + ' - ' + F.yunit
    });
    legendLineKeys(ctx.legend);

    var all = [], p, s, i, j;
    for (i = 0; i < F.panels.length; i++) {
      for (j = 0; j < F.panels[i].series.length; j++) {
        all = all.concat(F.panels[i].series[j].v);
      }
    }
    var dom = niceDomain(minOf(all), maxOf(all), 0.08, 5, true);
    var yt = linTicks(dom[0], dom[1], 5);
    var dec = tickDec(yt.step);

    var x0 = pad.l, x1 = pw - pad.r, y0 = ph - pad.b, y1 = pad.t;
    var inset = 18;
    var xAt = function (k) { return x0 + inset + k * ((x1 - x0 - 2 * inset) / (F.x.length - 1)); };
    var y = linScale(dom[0], dom[1], y0, y1);

    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      var svg = svgRoot(ctx.panels, pw, ph,
        'Multi-scale diffusion profile, ' + p.title,
        'Mean normalized response energy over 200 frames for t = 1, 4, 16, 32, three datasets, dense (solid) and sparse (dashed).');
      var k, tip;

      for (k = 0; k < yt.ticks.length; k++) {
        sLine(svg, x0, y(yt.ticks[k]), x1, y(yt.ticks[k]), 'gr');
        sTxt(svg, x0 - 6, y(yt.ticks[k]) + 3.2, fx(yt.ticks[k], dec), 't-tick', { 'text-anchor': 'end' });
      }
      sLine(svg, x0, y0, x1, y0, 'ax');
      sTxt(svg, x0, pad.t - 10, p.title, 't-pan');

      for (k = 0; k < F.x.length; k++) {
        sTxt(svg, xAt(k), y0 + 14, 't=' + F.x[k], 't-cat', { 'text-anchor': 'middle' });
      }

      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        var d = '', g = sEl('g', null, svg);
        for (k = 0; k < s.v.length; k++) {
          d += (k ? ' L' : 'M') + r2(xAt(k)) + ' ' + r2(y(s.v[k]));
        }
        sEl('path', {
          d: d, fill: 'none', stroke: s.color, 'stroke-width': 1.8,
          'stroke-linecap': 'round', 'stroke-linejoin': 'round',
          'stroke-dasharray': s.line === 'dashed' ? '5 3' : null
        }, g);
        for (k = 0; k < s.v.length; k++) {
          if (s.marker === 'square') {
            sRect(g, xAt(k) - 2.8, y(s.v[k]) - 2.8, 5.6, 5.6, { fill: s.color });
          } else {
            sEl('circle', { cx: r2(xAt(k)), cy: r2(y(s.v[k])), r: 3, fill: s.color }, g);
          }
          tip = s.name + '\nt = ' + F.x[k] + '\nq = ' + gf(s.v[k], 5) +
            (s.drd !== undefined ? '\nDRD-JS = ' + fx(s.drd, 4) : '');
          tipFor(sEl('circle', { cx: r2(xAt(k)), cy: r2(y(s.v[k])), r: 9, 'class': 'hit' }, g), tip);
        }
      }
    }

    var head = ['Signal / series'], rows = [];
    for (i = 0; i < F.x.length; i++) head.push('t=' + F.x[i]);
    head.push('DRD-JS');
    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        var row = [p.title + ' - ' + s.name];
        for (var m = 0; m < s.v.length; m++) row.push(gf(s.v[m], 5));
        row.push(s.drd !== undefined ? fx(s.drd, 4) : '-');
        rows.push(row);
      }
    }
    buildTable(ctx.table, 'Multi-scale diffusion profiles: mean normalized response energy per diffusion step (5 significant digits). Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderQProfiles(el, opts) {
    return run(el, 'qprofiles', opts, drawQProfiles);
  }

  /* ============================================ 2. rolloff / HF share ==== */

  function drawRolloffHfShare(mount, W, theme) {
    var F = DATA.rolloff;
    var cols = W >= 780 ? 2 : 1;
    var pw = panelWidth(W, cols, 14, 288, 560);
    var ph = 306;
    var pad = { l: 46, r: 12, t: 26, b: 56 };

    var ctx = scaffold(mount, {
      key: 'rolloff', cols: cols, theme: theme, panelMax: pw,
      axes: 'left - ' + F.panels[0].yname + ' · right - ' + F.panels[1].yname + ' · log y axis'
    });

    var dOrder = ['semantickitti', 'waymo', 'nuscenes'];
    var i, j, k, p, s;

    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      var vals = [];
      for (j = 0; j < p.series.length; j++) vals.push(p.series[j].dense, p.series[j].sparse);
      var dom = decadeDomain(minOf(vals), maxOf(vals), 2.2);
      var ticks = logTicks(dom[0], dom[1]);

      var svg = svgRoot(ctx.panels, pw, ph, p.title,
        'Grouped bars for three global signals and three datasets; dense bars are solid, sparse bars are hatched. Logarithmic y axis.');
      var defs = sEl('defs', null, svg);
      var hid = {};
      for (k = 0; k < dOrder.length; k++) {
        hid[dOrder[k]] = hatchDef(defs, DATA.datasets[dOrder[k]].color, uid('h'));
      }
      var x0 = pad.l, x1 = pw - pad.r, y0 = ph - pad.b, y1 = pad.t;
      var y = logScale(dom[0], dom[1], y0, y1);

      for (k = 0; k < ticks.length; k++) {
        sLine(svg, x0, y(ticks[k]), x1, y(ticks[k]), 'gr');
        sTxt(svg, x0 - 6, y(ticks[k]) + 3.2, gf(ticks[k], 3), 't-tick', { 'text-anchor': 'end' });
      }
      sLine(svg, x0, y0, x1, y0, 'ax');
      sTxt(svg, x0, pad.t - 10, p.title, 't-pan');

      var nSig = p.signals.length;
      var gw = (x1 - x0) / nSig;
      var gpad = gw * 0.09;
      var slotW = (gw - 2 * gpad) / 3;
      var barW = Math.max(3, slotW / 2 - 1.4);
      var showLab = barW >= 9;

      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        var si = p.signals.indexOf(s.signal);
        var dpos = 0;
        for (k = 0; k < dOrder.length; k++) if (dOrder[k] === s.ds) dpos = k;
        var slotX = x0 + si * gw + gpad + dpos * slotW;
        var sides = [
          { side: 'dense', v: s.dense, x: slotX },
          { side: 'sparse', v: s.sparse, x: slotX + barW + 1.4 }
        ];
        for (k = 0; k < 2; k++) {
          var bx = sides[k].x, bv = sides[k].v, by = y(bv);
          fillBar(svg, bx, by, barW, y0 - by, s.color, sides[k].side, hid[s.ds]);
          if (showLab) {
            sTxt(svg, bx + barW / 2 + 3, by - 4, gf(bv, 3), 't-val', {
              'text-anchor': 'start',
              transform: 'rotate(-90 ' + r2(bx + barW / 2 + 3) + ' ' + r2(by - 4) + ')'
            });
          }
          tipFor(sRect(svg, bx - 0.5, y1, barW + 1, y0 - y1, { 'class': 'hit' }),
            DATA.datasets[s.ds].display + ' · ' + s.signal + '\n' + sides[k].side + ' = ' + gf(bv, 5) +
            '\n' + (i === 0 ? 'lambda_85 (median over frames)' : 'P(lambda >= 0.6) (mean over frames)'));
        }
        if (slotW >= 26) {
          sTxt(svg, slotX + slotW / 2 - 0.7, y0 + 13, DATA.datasets[s.ds].short, 't-sub', { 'text-anchor': 'middle' });
        }
      }

      for (k = 0; k < nSig; k++) {
        var parts = String(p.signals[k]).split('_');
        var cx = x0 + (k + 0.5) * gw;
        var ly = y0 + (slotW >= 26 ? 27 : 16);
        for (var q = 0; q < parts.length; q++) {
          sTxt(svg, cx, ly + q * 11, parts[q], 't-cat', { 'text-anchor': 'middle' });
        }
        if (k) sLine(svg, x0 + k * gw, y0, x0 + k * gw, y0 + 4, 'ax');
      }
    }

    legendFillKeys(ctx.legend);

    var head = ['Panel', 'Signal', 'Dataset', 'Dense', 'Sparse', 'Sparse/dense'], rows = [];
    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        rows.push([p.key === 'left' ? 'lambda_85 rolloff' : 'HF share (lambda >= 0.6)',
          s.signal, DATA.datasets[s.ds].display, gf(s.dense, 3), gf(s.sparse, 3),
          gf(s.sparse / s.dense, 3)]);
      }
    }
    buildTable(ctx.table, 'Spectral roll-off and true high-frequency share, dense vs sparse (3 significant digits, as printed on the bars). Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderRolloffHfShare(el, opts) {
    return run(el, 'rolloff', opts, drawRolloffHfShare);
  }

  /* ============================================== 3. class retention ===== */

  function drawClassRetention(mount, W, theme, o) {
    var F = DATA.retention;
    var horizontal = W < 560;
    /* Default reproduces the published figure: the number above each bar is
       max(p75, median, HFR^(c)). opts.barLabel = 'median' prints the median
       instead, for the pending LOCAL call noted in the data README. */
    var useMedian = !!(o && o.barLabel === 'median');
    var labelOf = function (c) { return useMedian ? c.median : c.printed; };
    var ctx = scaffold(mount, {
      key: 'retention', cols: 1, theme: theme, panelMax: Math.min(W, 1180),
      axes: 'y: ' + F.yname + ' · ' + F.yunit + ' · bars = median, whiskers = IQR (p25-p75), black diamonds = full-spectrum HFR^(c)'
    });
    legendFillKeys(ctx.legend, false);
    var dk = hEl('span', 'lqc-key', ctx.legend);
    var dsvg = sEl('svg', { width: 22, height: 12, viewBox: '0 0 22 12', 'aria-hidden': 'true' }, dk);
    sEl('path', { d: 'M11 1 L15 6 L11 11 L7 6 Z', 'class': 'dia' }, dsvg);
    dk.appendChild(doc().createTextNode('HFR^(c) full-spectrum estimate'));
    if (useMedian) {
      ctx.note.appendChild(doc().createTextNode(
        ' This rendering prints the median instead of the published max(p75, median, HFR^(c)), because opts.barLabel = "median" was passed.'));
    }

    var i, j, c, s, maxPrinted = 0;
    for (i = 0; i < F.series.length; i++) {
      for (j = 0; j < F.series[i].classes.length; j++) {
        c = F.series[i].classes[j];
        var mx = Math.max(c.p75, c.median, c.diamond);
        if (mx > maxPrinted) maxPrinted = mx;
      }
    }
    var vmax = 1.22 * maxPrinted;
    var nCls = F.classes.length;
    var offs = [-0.26, 0, 0.26];

    var pw = Math.min(W, 1180), ph, pad, svg, x0, x1, y0, y1;

    if (!horizontal) {
      ph = 440;
      pad = { l: 44, r: 14, t: 28, b: 78 };
      svg = svgRoot(ctx.panels, pw, ph,
        'Per-class high-frequency retention at b0 = 16',
        'Grouped bars for seven coarse classes and three datasets: median retention with IQR whiskers, the full-spectrum HFR estimate as black diamonds, and the per-class frame count under each bar. Parity is 1.');
      x0 = pad.l; x1 = pw - pad.r; y0 = ph - pad.b; y1 = pad.t;
      var yt = linTicks(0, vmax, 5), dec = tickDec(yt.step);
      var y = linScale(0, vmax, y0, y1);
      for (i = 0; i < yt.ticks.length; i++) {
        sLine(svg, x0, y(yt.ticks[i]), x1, y(yt.ticks[i]), 'gr');
        sTxt(svg, x0 - 6, y(yt.ticks[i]) + 3.2, fx(yt.ticks[i], dec), 't-tick', { 'text-anchor': 'end' });
      }
      sLine(svg, x0, y0, x1, y0, 'ax');
      sLine(svg, x0, y(1), x1, y(1), 'par');
      sTxt(svg, x1 - 2, y(1) - 4.5, 'parity', 't-par', { 'text-anchor': 'end' });

      var pitch = (x1 - x0) / nCls;
      var barW = 0.2392 * pitch;
      for (j = 0; j < nCls; j++) {
        var cx = x0 + (j + 0.5) * pitch;
        for (i = 0; i < F.series.length; i++) {
          s = F.series[i];
          c = s.classes[j];
          var bx = cx + offs[i] * pitch - barW / 2;
          var by = y(c.median);
          fillBar(svg, bx, by, barW, y0 - by, s.color, 'dense', null, 0.9);
          var mid = bx + barW / 2;
          sLine(svg, mid, y(c.p25), mid, y(c.p75), 'wk');
          sLine(svg, mid - 2.6, y(c.p25), mid + 2.6, y(c.p25), 'wk');
          sLine(svg, mid - 2.6, y(c.p75), mid + 2.6, y(c.p75), 'wk');
          var dy = y(c.diamond);
          sEl('path', {
            d: 'M' + r2(mid) + ' ' + r2(dy - 3.8) + ' L' + r2(mid + 3.2) + ' ' + r2(dy) +
              ' L' + r2(mid) + ' ' + r2(dy + 3.8) + ' L' + r2(mid - 3.2) + ' ' + r2(dy) + ' Z',
            'class': 'dia'
          }, svg);
          var top = Math.min(by, y(c.p75), dy) - 5;
          if (barW >= 9) {
            sTxt(svg, mid + 3, top, fx(labelOf(c), 3), 't-val', {
              'text-anchor': 'start', transform: 'rotate(-90 ' + r2(mid + 3) + ' ' + r2(top) + ')'
            });
            sTxt(svg, mid + 3, y(0.03), 'n=' + c.n, 't-n', {
              'text-anchor': 'start', transform: 'rotate(-90 ' + r2(mid + 3) + ' ' + r2(y(0.03)) + ')'
            });
          }
          tipFor(sRect(svg, bx - 0.5, y1, barW + 1, y0 - y1, { 'class': 'hit' }),
            s.display + ' · ' + c.name + '\nmedian R = ' + fx(c.median, 4) +
            '\nIQR = ' + fx(c.p25, 4) + ' - ' + fx(c.p75, 4) +
            '\nHFR^(c) = ' + fx(c.diamond, 4) +
            '\nprinted label = ' + fx(c.printed, 3) + ' (max of p75, median, HFR)' +
            '\nn = ' + c.n + ' frames');
        }
        var words = String(F.classes[j]).split(' ');
        for (i = 0; i < words.length; i++) {
          sTxt(svg, cx, y0 + 16 + i * 11, words[i], 't-cat', { 'text-anchor': 'middle' });
        }
        if (j) sLine(svg, x0 + j * pitch, y0, x0 + j * pitch, y0 + 4, 'ax');
      }
    } else {
      var rowH = 15, groupH = rowH * 3 + 7;
      pad = { l: 100, r: 48, t: 24, b: 38 };
      ph = pad.t + nCls * groupH + pad.b;
      svg = svgRoot(ctx.panels, pw, ph,
        'Per-class high-frequency retention at b0 = 16',
        'Horizontal grouped bars for seven coarse classes and three datasets: median retention with IQR whiskers, the full-spectrum HFR estimate as black diamonds, and the per-class frame count. Parity is 1.');
      x0 = pad.l; x1 = pw - pad.r; y1 = pad.t; y0 = ph - pad.b;
      var x = linScale(0, vmax, x0, x1);
      var xt = linTicks(0, vmax, 4), xdec = tickDec(xt.step);
      for (i = 0; i < xt.ticks.length; i++) {
        sLine(svg, x(xt.ticks[i]), y1, x(xt.ticks[i]), y0, 'gr');
        sTxt(svg, x(xt.ticks[i]), y0 + 13, fx(xt.ticks[i], xdec), 't-tick', { 'text-anchor': 'middle' });
      }
      sLine(svg, x0, y1, x0, y0, 'ax');
      sLine(svg, x(1), y1, x(1), y0, 'par');
      sTxt(svg, x(1), y1 - 6, 'parity', 't-par', { 'text-anchor': 'middle' });
      for (j = 0; j < nCls; j++) {
        var gy = pad.t + j * groupH;
        sTxt(svg, x0 - 7, gy + groupH / 2 - 2, F.classes[j], 't-cat', { 'text-anchor': 'end' });
        for (i = 0; i < F.series.length; i++) {
          s = F.series[i];
          c = s.classes[j];
          var by2 = gy + i * rowH + 2.5, bh = rowH - 4.5;
          fillBar(svg, x0, by2, x(c.median) - x0, bh, s.color, 'dense', null, 0.9);
          var mid2 = by2 + bh / 2;
          sLine(svg, x(c.p25), mid2, x(c.p75), mid2, 'wk');
          sLine(svg, x(c.p25), mid2 - 2.6, x(c.p25), mid2 + 2.6, 'wk');
          sLine(svg, x(c.p75), mid2 - 2.6, x(c.p75), mid2 + 2.6, 'wk');
          var dx = x(c.diamond);
          sEl('path', {
            d: 'M' + r2(dx) + ' ' + r2(mid2 - 3.6) + ' L' + r2(dx + 3.1) + ' ' + r2(mid2) +
              ' L' + r2(dx) + ' ' + r2(mid2 + 3.6) + ' L' + r2(dx - 3.1) + ' ' + r2(mid2) + ' Z',
            'class': 'dia'
          }, svg);
          sTxt(svg, Math.max(x(c.printed), x(c.p75)) + 5, mid2 + 3, fx(labelOf(c), 3), 't-val', { 'text-anchor': 'start' });
          sTxt(svg, x0 + 3, mid2 + 3, 'n=' + c.n, 't-n', { 'text-anchor': 'start' });
          tipFor(sRect(svg, x0, by2 - 1, x1 - x0, bh + 2, { 'class': 'hit' }),
            s.display + ' · ' + c.name + '\nmedian R = ' + fx(c.median, 4) +
            '\nIQR = ' + fx(c.p25, 4) + ' - ' + fx(c.p75, 4) +
            '\nHFR^(c) = ' + fx(c.diamond, 4) +
            '\nprinted label = ' + fx(c.printed, 3) + ' (max of p75, median, HFR)' +
            '\nn = ' + c.n + ' frames');
        }
      }
    }

    var head = ['Dataset', 'Class', 'Printed label', 'Median R', 'p25', 'p75', 'HFR^(c)', 'n'], rows = [];
    for (i = 0; i < F.series.length; i++) {
      s = F.series[i];
      for (j = 0; j < s.classes.length; j++) {
        c = s.classes[j];
        rows.push([s.display, c.name, fx(c.printed, 3), fx(c.median, 3), fx(c.p25, 3),
          fx(c.p75, 3), fx(c.diamond, 3), String(c.n)]);
      }
    }
    buildTable(ctx.table, 'Per-class HF retention at b0 = 16. "Printed label" is the value drawn above each bar, max(p75, median, HFR^(c)); the median is the bar height. Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderClassRetention(el, opts) {
    return run(el, 'retention', opts, drawClassRetention);
  }

  /* ================================================== 4. gap heat map ==== */

  function drawGapHeatmap(mount, W, theme) {
    var F = DATA.heatmap;
    var cols = gridCols(W, 1000, 640);
    var pw = panelWidth(W, cols, 14, 300, 620);
    var nCol = F.columns.length;
    var labelW = cols === 1 ? 108 : 94;
    var pad = { l: labelW, r: 10, t: 26, b: 48 };
    var cellW = Math.min((pw - pad.l - pad.r) / nCol, 46);
    var cellH = cols === 1 ? 24 : 22;
    var gridW = cellW * nCol;
    var vmax = F.vmax;

    var ctx = scaffold(mount, {
      key: 'heatmap', cols: cols, theme: theme, panelMax: Math.ceil(pad.l + gridW + pad.r),
      axes: 'x: ' + F.xname + ' · y: ' + F.yname + ' · ' + F.meaning
    });
    legendColorbar(ctx.legend, vmax);

    var order = ['semantickitti', 'waymo', 'nuscenes'];
    var di, ds, rowsData, i, j, svg, x0, y1, rr, v, cx, cy, txt;
    var showText = cellW >= 22;
    var oneDecimal = cellW >= 30;
    var fs = Math.min(8.4, Math.max(6.6, cellW * 0.285));

    for (di = 0; di < order.length; di++) {
      ds = order[di];
      rowsData = F.matrices[ds];
      if (!rowsData) continue;
      var ph = pad.t + rowsData.length * cellH + pad.b;
      var pwx = pad.l + gridW + pad.r;
      svg = svgRoot(ctx.panels, pwx, ph,
        'Dense-minus-sparse band-energy gap, ' + DATA.datasets[ds].display,
        'Heat map of the mean per-band energy gap G = P_dense - P_sparse for three global signals and seven classes across ten graph-frequency bands. Red is dense excess, blue is sparse excess, diverging scale symmetric about zero with vmax ' + gf(vmax, 3) + '.');
      x0 = pad.l; y1 = pad.t;
      sTxt(svg, 0, pad.t - 10, DATA.datasets[ds].display, 't-pan', { fill: DATA.datasets[ds].color });

      for (i = 0; i < rowsData.length; i++) {
        rr = rowsData[i];
        cy = y1 + i * cellH;
        var nameOnly = String(rr.row).replace(/\s*\(n=\d+\)\s*$/, '');
        var lab = sTxt(svg, x0 - 7, cy + cellH / 2 + 3, '', 't-cat', { 'text-anchor': 'end' });
        sSpan(lab, nameOnly);
        if (labelW >= 104 && rr.kind === 'class') sSpan(lab, ' n=' + rr.n, 't-dim');
        for (j = 0; j < nCol; j++) {
          v = rr.v[j];
          cx = x0 + j * cellW;
          sRect(svg, cx, cy, cellW, cellH, {
            fill: (v === null || !isFinite(v)) ? 'none' : rdbuR((v + vmax) / (2 * vmax)),
            stroke: 'none'
          });
          sRect(svg, cx, cy, cellW, cellH, { fill: 'none', 'class': 'cellsep' });
          if (v === null || !isFinite(v)) {
            sTxt(svg, cx + cellW / 2, cy + cellH / 2 + 3, 'N/A', 't-cell', {
              'text-anchor': 'middle', fill: '#666666', 'font-size': fs
            });
          } else if (showText) {
            txt = oneDecimal ? fx(v * 1000, 1) : fx(v * 1000, 0);
            sTxt(svg, cx + cellW / 2, cy + cellH / 2 + 3, txt, 't-cell', {
              'text-anchor': 'middle',
              fill: Math.abs(v) > 0.55 * vmax ? '#ffffff' : '#222222',
              'font-size': fs
            });
          }
          tipFor(sRect(svg, cx, cy, cellW, cellH, { 'class': 'hit' }),
            DATA.datasets[ds].display + ' · ' + nameOnly +
            '\nlambda band ' + F.columns[j] +
            '\nG x 1000 = ' + (v === null || !isFinite(v) ? 'N/A' : sgn(v * 1000, 1)) +
            '\nG = ' + (v === null || !isFinite(v) ? 'N/A' : sgn(v, 6)) +
            '\nn = ' + rr.n + ' frames');
        }
      }
      sRect(svg, x0, y1, gridW, rowsData.length * cellH, { fill: 'none', 'class': 'ax' });

      var step = cellW >= 30 ? 1 : 2;
      for (j = 0; j < nCol; j += step) {
        var lx = x0 + (j + 0.5) * cellW, ly = y1 + rowsData.length * cellH + 11;
        sTxt(svg, lx, ly, F.columns[j], 't-dim', {
          'text-anchor': 'end', transform: 'rotate(-45 ' + r2(lx) + ' ' + r2(ly) + ')'
        });
      }
    }

    var head = ['Dataset', 'Row', 'n'], rows = [], k;
    for (j = 0; j < nCol; j++) head.push(F.columns[j]);
    for (di = 0; di < order.length; di++) {
      ds = order[di];
      rowsData = F.matrices[ds] || [];
      for (i = 0; i < rowsData.length; i++) {
        rr = rowsData[i];
        var row = [DATA.datasets[ds].display, String(rr.row).replace(/\s*\(n=\d+\)\s*$/, ''), String(rr.n)];
        for (k = 0; k < nCol; k++) {
          row.push(rr.v[k] === null || !isFinite(rr.v[k]) ? 'N/A' : fx(rr.v[k] * 1000, 1));
        }
        rows.push(row);
      }
    }
    buildTable(ctx.table, 'Mean band-energy gap G = P_dense - P_sparse, x 1000, merged 10-column variant (bands 1+2 merged inside the Jackson kernel soft edge). Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderGapHeatmap(el, opts) {
    return run(el, 'heatmap', opts, drawGapHeatmap);
  }

  /* =================================================== 5. b0 trend ======= */

  function drawB0Trend(mount, W, theme) {
    var F = DATA.b0trend;
    var cols = gridCols(W, 980, 640);
    var pw = panelWidth(W, cols, 14, 264, 420);
    var ph = 252;
    var pad = { l: 42, r: 14, t: 26, b: 34 };

    var ctx = scaffold(mount, {
      key: 'b0trend', cols: cols, theme: theme, panelMax: pw,
      axes: 'x: ' + F.xname + ' (' + F.xscale + ') · y: ' + F.yname + ' · ' + F.yunit
    });
    legendClassKeys(ctx.legend);

    var all = [], i, j, k, p, s;
    for (i = 0; i < F.panels.length; i++) {
      for (j = 0; j < F.panels[i].series.length; j++) all = all.concat(F.panels[i].series[j].v);
    }
    var dom = niceDomain(minOf(all), maxOf(all), 0.06, 5, false);
    var yt = linTicks(dom[0], dom[1], 5), dec = tickDec(yt.step);
    var xlo = F.x[0] / 1.25, xhi = F.x[F.x.length - 1] * 1.25;

    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      var svg = svgRoot(ctx.panels, pw, ph,
        'Per-class HF retention against the diffusion cutoff, ' + p.title,
        'Seven class lines of median high-frequency retention for diffusion cutoffs b0 = 2, 8, 16, 32 on a log-2 x axis; the dashed line marks parity.');
      var x0 = pad.l, x1 = pw - pad.r, y0 = ph - pad.b, y1 = pad.t;
      var x = logScale(xlo, xhi, x0, x1);
      var y = linScale(dom[0], dom[1], y0, y1);

      for (k = 0; k < yt.ticks.length; k++) {
        sLine(svg, x0, y(yt.ticks[k]), x1, y(yt.ticks[k]), 'gr');
        sTxt(svg, x0 - 6, y(yt.ticks[k]) + 3.2, fx(yt.ticks[k], dec), 't-tick', { 'text-anchor': 'end' });
      }
      sLine(svg, x0, y0, x1, y0, 'ax');
      sLine(svg, x0, y(1), x1, y(1), 'par');
      sTxt(svg, x1 - 2, y(1) - 4.5, 'parity', 't-par', { 'text-anchor': 'end' });
      sTxt(svg, x0, pad.t - 10, p.title, 't-pan', { fill: p.color });

      for (k = 0; k < F.x.length; k++) {
        sTxt(svg, x(F.x[k]), y0 + 14, String(F.x[k]), 't-cat', { 'text-anchor': 'middle' });
        sLine(svg, x(F.x[k]), y0, x(F.x[k]), y0 + 4, 'ax');
      }

      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        var g = sEl('g', null, svg), d = '';
        for (k = 0; k < s.v.length; k++) d += (k ? ' L' : 'M') + r2(x(F.x[k])) + ' ' + r2(y(s.v[k]));
        sEl('path', {
          d: d, fill: 'none', stroke: s.color, 'stroke-width': 1.8,
          'stroke-linecap': 'round', 'stroke-linejoin': 'round'
        }, g);
        for (k = 0; k < s.v.length; k++) {
          sEl('circle', { cx: r2(x(F.x[k])), cy: r2(y(s.v[k])), r: 2.7, fill: s.color }, g);
          tipFor(sEl('circle', { cx: r2(x(F.x[k])), cy: r2(y(s.v[k])), r: 9, 'class': 'hit' }, g),
            p.title + ' · ' + s.name + '\nb0 = ' + F.x[k] + '\nmedian R = ' + fx(s.v[k], 4) +
            '\nn = ' + s.n + ' frames');
        }
      }
    }

    var head = ['Dataset', 'Class', 'n'], rows = [];
    for (i = 0; i < F.x.length; i++) head.push('b0=' + F.x[i]);
    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      for (j = 0; j < p.series.length; j++) {
        s = p.series[j];
        var row = [p.title, s.name, String(s.n)];
        for (k = 0; k < s.v.length; k++) row.push(fx(s.v[k], 4));
        rows.push(row);
      }
    }
    buildTable(ctx.table, 'Median per-class HF retention against the diffusion cutoff b0 (4 decimals). Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderB0Trend(el, opts) {
    return run(el, 'b0trend', opts, drawB0Trend);
  }

  /* ================================================ 6. band profiles ===== */

  function drawBandProfiles(mount, W, theme) {
    var F = DATA.bands;
    var cols = gridCols(W, 980, 640);
    var pw = panelWidth(W, cols, 14, 268, 400);
    var ph = 226;
    var pad = { l: 40, r: 10, t: 26, b: 46 };
    var nB = F.bands.length;

    var ctx = scaffold(mount, {
      key: 'bands', cols: cols, theme: theme, panelMax: pw,
      axes: 'x: ' + F.xname + ' (11 bands) · y: ' + F.yname + ' · ' + F.yunit + ' · log y axis'
    });

    var i, j, k, p, s;
    var rowDom = {};
    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      var acc = rowDom[p.ds] || [];
      for (j = 0; j < p.series.length; j++) acc = acc.concat(p.series[j].v);
      rowDom[p.ds] = acc;
    }
    var dom = {};
    for (k in rowDom) {
      if (rowDom.hasOwnProperty(k)) {
        dom[k] = [Math.pow(10, Math.floor(log10(minOf(rowDom[k])) + 1e-9)),
          Math.pow(10, Math.ceil(log10(maxOf(rowDom[k])) - 1e-9))];
      }
    }

    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      var d = dom[p.ds];
      var svg = svgRoot(ctx.panels, pw, ph,
        'Band-energy profile, ' + DATA.datasets[p.ds].display + ', ' + p.signal,
        'Eleven graph-frequency bands; mean per-band energy fraction over 200 frames, dense (solid fill) against sparse (hatched fill), logarithmic y axis.');
      var defs = sEl('defs', null, svg);
      var hid = hatchDef(defs, p.color, uid('h'));
      var x0 = pad.l, x1 = pw - pad.r, y0 = ph - pad.b, y1 = pad.t;
      var y = logScale(d[0], d[1], y0, y1);
      var ticks = logTicks(d[0], d[1]);

      for (k = 0; k < ticks.length; k++) {
        sLine(svg, x0, y(ticks[k]), x1, y(ticks[k]), 'gr');
        sTxt(svg, x0 - 5, y(ticks[k]) + 3.2, gf(ticks[k], 3), 't-tick', { 'text-anchor': 'end' });
      }
      sLine(svg, x0, y0, x1, y0, 'ax');
      var title = sTxt(svg, x0, pad.t - 10, '', 't-pan');
      sSpan(title, DATA.datasets[p.ds].display).setAttribute('fill', p.color);
      sSpan(title, '  ' + p.signal, 't-dim');

      var pitch = (x1 - x0) / nB;
      var barW = Math.max(2, pitch / 2 - 1.1);
      for (j = 0; j < nB; j++) {
        var gx = x0 + j * pitch + (pitch - 2 * barW - 1) / 2;
        for (k = 0; k < p.series.length; k++) {
          s = p.series[k];
          var bv = s.v[j], by = y(bv), bx = gx + k * (barW + 1);
          fillBar(svg, bx, by, barW, y0 - by, p.color, s.side, hid);
          tipFor(sRect(svg, bx - 0.5, y1, barW + 1, y0 - y1, { 'class': 'hit' }),
            DATA.datasets[p.ds].display + ' · ' + p.signal +
            '\nlambda band ' + F.bands[j] + '\n' + s.name + ' = ' + gf(bv, 4));
        }
      }
      sLine(svg, x0 + 2 * pitch, y1, x0 + 2 * pitch, y0, 'soft');

      var step = pitch >= 22 ? 1 : 2;
      for (j = 0; j < nB; j += step) {
        var lx = x0 + (j + 0.5) * pitch, ly = y0 + 10;
        sTxt(svg, lx, ly, F.bands[j], 't-dim', {
          'text-anchor': 'end', transform: 'rotate(-45 ' + r2(lx) + ' ' + r2(ly) + ')'
        });
      }
    }

    legendFillKeys(ctx.legend, null);
    var sk = hEl('span', 'lqc-key', ctx.legend);
    var ssvg = sEl('svg', { width: 22, height: 12, viewBox: '0 0 22 12', 'aria-hidden': 'true' }, sk);
    sEl('line', { x1: 11, y1: 0, x2: 11, y2: 12, 'class': 'soft' }, ssvg);
    sk.appendChild(doc().createTextNode('Jackson kernel soft edge (lambda = 0.03)'));

    var head = ['Dataset', 'Signal', 'Side'], rows = [];
    for (j = 0; j < nB; j++) head.push(F.bands[j]);
    for (i = 0; i < F.panels.length; i++) {
      p = F.panels[i];
      for (k = 0; k < p.series.length; k++) {
        s = p.series[k];
        var row = [DATA.datasets[p.ds].display, p.signal, s.name];
        for (j = 0; j < s.v.length; j++) row.push(gf(s.v[j], 4));
        rows.push(row);
      }
    }
    buildTable(ctx.table, 'Mean per-band energy fraction over 11 graph-frequency bands, dense and sparse (4 significant digits). Run ' + DATA.meta.run + '.', head, rows);
    return ctx.fig;
  }

  function renderBandProfiles(el, opts) {
    return run(el, 'bands', opts, drawBandProfiles);
  }

  /* ---------------------------------------------------------------- public */

  var KEYS = {
    'qprofiles': renderQProfiles,
    'rolloff-hfshare': renderRolloffHfShare,
    'class-retention': renderClassRetention,
    'gap-heatmap': renderGapHeatmap,
    'b0-trend': renderB0Trend,
    'band-profiles': renderBandProfiles
  };

  function autoInit(opts) {
    var D = doc(), out = {}, i, n, k;
    if (!D || typeof D.querySelectorAll !== 'function') return out;
    var nodes = D.querySelectorAll('[data-lieq-chart]');
    for (i = 0; i < nodes.length; i++) {
      n = nodes[i];
      k = n.getAttribute('data-lieq-chart');
      if (KEYS[k]) out[k] = KEYS[k](n, opts);
    }
    return out;
  }

  function renderAll(map, opts) {
    if (!map) return autoInit(opts);
    var out = {}, k;
    for (k in map) {
      if (map.hasOwnProperty(k) && KEYS[k]) out[k] = KEYS[k](map[k], opts);
    }
    return out;
  }

  var API = {
    version: '1.0.0',
    data: DATA,
    keys: ['qprofiles', 'rolloff-hfshare', 'class-retention', 'gap-heatmap', 'b0-trend', 'band-profiles'],
    renderQProfiles: renderQProfiles,
    renderRolloffHfShare: renderRolloffHfShare,
    renderClassRetention: renderClassRetention,
    renderGapHeatmap: renderGapHeatmap,
    renderB0Trend: renderB0Trend,
    renderBandProfiles: renderBandProfiles,
    renderAll: renderAll,
    autoInit: autoInit,
    noteText: noteText,
    _internals: { gf: gf, rdbuR: rdbuR, logTicks: logTicks, linTicks: linTicks, injectCSS: injectCSS }
  };

  /* Own, unambiguous global. The sibling model-comparison module on this page also
     publishes window.LiEQCharts, so we never overwrite it: we fill in only the keys it
     has not already defined, and whichever script loads last leaves both reachable. */
  global.LiEQSpectralCharts = API;
  var NS = global.LiEQCharts;
  if (!NS || typeof NS !== 'object') {
    NS = {};
    global.LiEQCharts = NS;
  }
  for (var ak in API) {
    if (API.hasOwnProperty(ak) && !Object.prototype.hasOwnProperty.call(NS, ak)) {
      NS[ak] = API[ak];
    }
  }
  if (typeof module === 'object' && module && module.exports) module.exports = API;

  var D0 = doc();
  if (D0 && D0.addEventListener) {
    if (D0.readyState === 'loading') {
      D0.addEventListener('DOMContentLoaded', function () { autoInit(); });
    } else {
      autoInit();
    }
  }
}(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this)));
