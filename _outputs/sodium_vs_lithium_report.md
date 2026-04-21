# Sodium-ion vs Lithium-ion Batteries for Electric Vehicles: Comparative Report (2018–2026)

Executive summary

Sodium-ion batteries (SIBs) are an emerging, lower-cost battery chemistry using abundant sodium and iron/manganese-based active materials. For EVs, SIBs offer safety and low-temperature advantages and materially lower reliance on critical lithium minerals, but they currently lag lithium-ion (Li-ion) in specific energy and volumetric density. Key findings:
- Typical cell-level specific energy: Li-ion (cell) ≈ 150–300 Wh/kg; SIB (cell) ≈ 100–175 Wh/kg (confidence: medium). CATL and others have announced SIB cells up to ≈175 Wh/kg (company claims).
- Cycle life: SIBs can achieve >1,000 cycles in recent demonstrations; some pouch prototypes report >1,200 cycles with ~88% retention (confidence: medium).
- Cost: Prospective cell-material cost for mature SIBs is projected to be substantially lower than Li-ion in many scenarios; long-run estimates range down toward $40–95/kWh (low–medium confidence, scenario-dependent). Li-ion pack prices in 2024 are typically around $100–150/kWh (pack), with cell costs higher (confidence: medium).
- Supply chain & geopolitics: Sodium feedstocks are globally abundant and widely distributed; lithium, cobalt, nickel, and graphite supply chains are more geographically concentrated, with refining concentrated in China (confidence: high for geographic concentration).
- LCA: Recent prospective LCAs (Wickerts et al., 2024) model SIB cradle-to-gate GHG impacts in the range ≈58–130 kg CO2-eq per kWh (scenario dependent); SIBs show lower mineral scarcity impacts vs NMC-type LIBs (confidence: medium).

Recommendation (short): SIBs are best targeted to cost-sensitive, space-tolerant EV segments (city cars, two- and three-wheelers, light commercial vehicles, buses, and stationary storage) where lower energy density is acceptable and low-temperature performance or safety is valuable. Li-ion (NMC, NCA, LFP) remains preferable for long-range passenger cars, trucks, and applications where volumetric/gravimetric energy density is critical.

Table of contents

- Executive summary
- Energy & power metrics
- Materials & chemistry
- Manufacturing & scale-up
- Supply chain, critical minerals & geopolitics
- Market players & recent announcements (2020–2026)
- Lifecycle environmental trade-offs (LCA)
- Cost comparison and trajectories
- Use-cases and recommendations
- Uncertainties and data gaps
- References

---

1) Energy & power metrics

Summary table (typical ranges; cell-level unless noted)

| Metric | Sodium-ion (SIB) | Lithium-ion (LIB) | Confidence |
|---|---:|---:|---|
| Specific energy (cell, Wh/kg) | 100–175 (industry/TRL demonstrations 120–160 typical; CATL claims up to 175) | 150–300 (LFP ~120–170; high-Ni NMC up to 250–300 in cells) | Medium |
| Volumetric energy (Wh/L, cell) | ~200–400 (device-dependent) | ~300–700 (device-dependent) | Low–Medium |
| Pack-level specific energy (Wh/kg) | ~90–150 (early commercial packs lower due to engineering margin) | ~140–260 (varies by chemistry and pack design) | Medium |
| Power density (W/kg) | Good — SIBs support high-rate charge/discharge; power density comparable to mid-power Li-ion | Wide range; high-power NMC/LFP designs exceed SIB in many cases | Medium |
| Typical cycle life (cells) | 1,000–2,000+ cycles (recent pouch demonstrations report >1,200 cycles with ≈88% retention after 900–1,200 cycles) | 1,000–3,000+ (LFP commonly >3,000; high-energy NMC ~1,000–2,000) | Medium |
| Calendar life | Several years (projected comparable to LFP in favourable designs) | Several years; LFP & advanced formulations show long calendar life | Low–Medium |
| Temperature performance | Better low-temperature performance reported by some vendors (CATL claims improved cold-start/winter behaviour) | Li-ion performance degrades in cold; some formats engineered for cold (thermal management) | Medium |

Notes & sources: cell-level SIB modelling in LCA studies commonly uses ~160 Wh/kg as a plausible mature SIB cell (Wickerts et al., 2024). Company claims (CATL) report up to ≈175 Wh/kg for commercial Naxtra cells (CATL, 2025). Li-ion ranges reflect typical LFP and NMC family values reported in industry reviews and IEA analyses (IEA, 2024; Thunder Said Energy analysis).

2) Materials & chemistry

Key differences
- Cathodes: SIBs typically use layered transition metal oxides (Na layered oxides: NaM oxides where M = Mn, Fe, Ni partial), polyanionic compounds or Prussian-blue analogues. LIB cathodes include NMC, NCA, LFP, etc., often with nickel, cobalt, manganese (NMC/NCA) or iron-phosphate (LFP).
- Anodes: SIBs generally use hard carbon (HC) as the commercial anode; graphite intercalation is not viable for Na+ without special treatments. LIB anodes use graphite (and silicon blends in high-energy cells).
- Electrolyte: Many SIB electrolytes reuse organic carbonate solvents with sodium salts (NaPF6, NaClO4 variants), although SEI chemistry and additives differ from Li-ion; Na+ forms different interphases and can lead to larger initial irreversible capacity loss (first-cycle loss) in some HC materials.

Material intensity & raw inputs (illustrative — kg per kWh; see IRENA/IRENA-2024 dataset for modelling assumptions)

| Material (kg/kWh) | Typical LIB (e.g., LFP/NMC mix) | Representative SIB (modeled) | Note |
|---|---:|---:|---|
| Lithium (kg/kWh, Li metal equiv.) | ~0.4–0.6 kg Li per kWh (varies by chemistry; IRENA reported ~0.53 kg/kWh for LFP-type) | 0 (sodium-based) | IRENA material composition table (see sources) |
| Nickel / Cobalt / Manganese (kg/kWh) | NMC/NCA chemistries contain 0.1–0.7 kg of transition metals per kWh depending on cathode | SIB uses Mn/Fe in layered oxides; lower Ni/Co dependency | SIBs may avoid critical Co/Ni entirely in many designs |
| Graphite (anode) | ~0.8–1 kg/kWh (varies) | Hard carbon ~0.9–1.2 kg/kWh (hard carbon precursor costs differ) | Hard carbon often made from biomass, petrocarbon precursors; process impacts important |
| Sodium (as salt) | 0 | small but abundant (Na2CO3 precursors inexpensive) | Sodium raw material cost and availability not constraining |

Performance-limiting mechanisms
- Lower cell voltage and larger ionic radius of Na+ reduce theoretical gravimetric energy vs Li+. SIBs therefore require materials engineering (high-capacity cathodes, optimized hard carbon) to approach Li-ion cell energies.
- First-cycle irreversible capacity (ICE) and SEI formation on hard carbon is an engineering challenge for SIB full cells; electrolyte additives and electrode pre-treatment reduce ICE.
- Cathode structural degradation (layered oxide phase changes) and transition-metal dissolution are aging paths for SIBs similar to LIBs but with different kinetics due to Na+ size.

Sources: RSC 2024 review on hard carbon; IRENA technology brief 2025; Chalmers LCA (Wickerts et al., 2024).

3) Manufacturing & scale-up

Process differences
- Many cell manufacturing steps are shared (coating, calendaring, cell assembly, formation, aging, module/pack integration). Differences center on electrode formulation (active materials, slurries) and electrolyte salt sourcing.
- Active material synthesis: Prussian-blue analogues and large-scale hard carbon manufacture require scaled precursor synthesis; CATL and others have invested in large-scale Prussian-white production to support SIB capacity.

Scale-up and plants
- China leads early commercial SIB manufacturing activity and announced projects are large (e.g., a reported 20 GWh SIB plant project announced in late 2025). CATL announced the Naxtra platform and mass-production roadmaps in 2025 and early 2026 (CATL press release and industry reporting).
- Typical timeframe to scale to GWh/year: announcements and pilot lines in 2024–2026 suggest multi-year ramp (2–4 years) to reach >1–10 GWh/year per facility depending on capital and policy support.

Capital intensity & manufacturing cost sensitivity
- Upgrading or repurposing Li-ion gigafactories for SIB production is feasible in many cases, lowering incremental capital intensity. Key capital needs include upstream active-material plants (Prussian white, hard carbon) and formation/aging lines.

Sources: PV Magazine (2025), CATL (2025), Sandia (DOE) project summaries.

4) Supply chain, critical minerals & geopolitics

Geographic distribution & concentration risks
- Lithium: mined mainly in Australia, Chile, Argentina; refining concentrated in China for carbonate/hydroxide and graphite processing (IEA, 2024).
- Cobalt, nickel: supply concentrated in a handful of countries; cobalt notably exposed to concentration and social risk in DRC.
- Sodium: feedstock (sodium salts) is globally abundant and widely distributed—low concentration risk.
- Iron & manganese (used in many SIB cathodes) are widely available globally.

Recyclability & secondary supply
- Existing recycling value is driven by high-value metals (Li, Co, Ni, Cu). SIB chemistries with lower Ni/Co content reduce material value; recycling will still be technically feasible but economics may differ—policy and regulatory frameworks (EPR, recycling mandates) will influence secondary supply development.

Sources: IEA Global Critical Minerals Outlook (2025), IRENA (2025), Argonne/US DOE reports.

5) Market and who is leading (companies, research groups, countries)

Major companies & actors advancing SIBs
- CATL (China): Naxtra sodium-ion platform; public roadmap for mass production and vehicle deployments (2025–2026). [CATL news]
- BYD, EVE Energy, CALB and other Chinese manufacturers: active in alternative chemistries and SIB R&D and pilot projects (industry reporting).
- Startups and research groups: Altris (Sweden) (iron–sodium chemistries), academic groups in China/Europe (hard carbon, Prussian-white research).
- National research & labs: Sandia (US DOE-funded projects), Chalmers (Sweden) LCA and materials work.

Recent developments (2024–2026)
- CATL (2025–2026): launched the Naxtra sodium-ion platform; announced mass-production roadmap and supplier communications and claimed cell energy up to ~175 Wh/kg (CATL news, 2025).
- Guangde Qingna Technology Co. (Nov 2025): announced a 20 GWh sodium-ion manufacturing project in Sichuan province (PV Magazine, 24 Nov 2025).
- 2024–2026: multiple pilot integrations and early commercial SIB deployments reported in China; industry analysts project tens to hundreds of GWh pipeline capacity by 2030 in scenarios where SIBs scale (industry press, IRENA summary).

Confidence: medium for company activity (well-documented in industry press), low–medium for exact production volumes until factories enter production.

6) Lifecycle environmental trade-offs (LCA)

Key LCA findings (select studies)
- Prospective cradle-to-gate LCAs of SIBs (Wickerts et al., Chalmers, 2024) model SIB cells at ~160 Wh/kg and report cradle-to-gate GHG impacts in the range ≈58–130 kg CO2-eq/kWh depending on material choices and electricity mix. The studies find lower mineral resource scarcity impacts for SIBs vs NMC LIBs due to the absence of critical Ni/Co and Li demand.
- Comparative studies show production-stage impacts depend strongly on specific energy (Wh/kg): lower energy density increases per-kWh material requirements and can raise per-kWh GHG impacts unless materials and processing are low-carbon.

Water use & local impacts
- Lithium brine extraction (e.g., South America) raises concerns over local water use and community impacts; hard-rock lithium mining and refining have different local footprints. Sodium salt extraction and iron/manganese mining are generally less water intensive per unit of produced active material and geographically widespread.

Recyclability
- Lower-value SIB chemistries may have less immediate economic incentive for metal recovery compared with high-value NMC materials; regulation and extended producer responsibility can shift economics.

Sources: Wickerts et al., 2024; Guo et al., 2023; IRENA brief (2025).

7) Cost: cell-level and trajectories

Typical 2024–2026 estimates
- Li-ion: industry pack prices broadly quoted in the $100–150/kWh (pack) range in 2024; cell-level costs higher depending on chemistry and configuration (IEA/BNEF summaries).
- SIB: material- and cell-cost modelling (BatPaC-style scenarios and projections) suggest mature SIB cells could reach material-level costs roughly in the $40–95/kWh range in favourable scenarios; current prototype/material-cost statements (Sandia/DOE project notes, IRENA analysis) show higher near-term costs and strong sensitivity to hard carbon and cathode production costs.

Price sensitivity
- SIB competitiveness depends mainly on realized specific energy (Wh/kg) and hard-carbon costs; if SIB cell energy approaches ~160–180 Wh/kg at low material cost, pack-level competitiveness for city EVs and two-wheelers becomes attractive.

Confidence: low–medium (many estimates are scenario-based and depend on future scaling and raw-material prices).

8) Use-cases & recommendations

Best-fit EV segments for SIBs
- City cars and short-range passenger vehicles where lower range is acceptable in exchange for lower cost and better cold-weather performance.
- Two- and three-wheelers, light commercial vehicles (last-mile delivery) where lower upfront cost and high cycle life/reliability are valuable.
- Buses and some heavy-duty applications where energy density trade-offs are acceptable for lower cost and safety advantages.
- Stationary storage and hybrid pack strategies (dual-power architectures mixing SIB & Li-ion) to exploit cost advantages while preserving range-critical Li-ion for energy-dense zones (CATL dual-power concept).

Where Li-ion remains preferable
- Long-range BEVs, heavy-duty trucks, aviation, and any application where mass/volume are highly constrained.

Research & policy recommendations
- Fund scale-up pilots and independent LCA studies based on factory-scale inventories for SIB materials and cells (reduce data gaps).
- Support recycling infrastructure that accommodates lower-Ni/Co chemistries to ensure circularity as SIB deployment grows.
- Encourage standards and cell-testing regimes for SIBs (safety, low-temperature performance, calendar life) so automakers can confidently integrate cells.

9) Uncertainties and data gaps

- Many SIB numbers are prospectively modelled, not measured at gigafactory scale; lab-to-factory translation uncertainty is high (confidence: low–medium).
- Company performance claims (e.g., CATL energy-density figures) need independent verification in third-party cell tests (confidence: low without test reports).
- LCA sensitivity: results vary greatly with assumed specific energy, electricity mix for material production, and end-of-life assumptions (confidence: medium).
- Recycling economics for low-Ni/Co SIB chemistries are not well-established; limited commercial recycling experience exists (confidence: low).

10) References (selected primary sources; numbered)

1. Wickerts S., Arvidsson R., Nordelöf A., et al. (2024) "Prospective life cycle assessment of sodium-ion batteries made from abundant elements." Journal of Industrial Ecology. DOI: 10.1111/jiec.13452. https://research.chalmers.se/publication/538534/file/538534_Fulltext.pdf

2. IRENA (2025) "Sodium-ion batteries: A technology brief." International Renewable Energy Agency. https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Nov/IRENA_TEC_Sodium-ion_batteries_2025.pdf

3. CATL (2025) "Naxtra Battery Breakthrough & Dual-Power Architecture." CATL news release. https://www.catl.com/en/news/6401.html

4. RSC (2024) Chun Wu et al., "Hard carbon for sodium-ion batteries: progress, strategies and future perspective." Chemical Science (RSC), 2024. https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc00734d

5. PV Magazine (2025) "Massive 20 GWh sodium-ion battery manufacturing plant announced in China." https://www.pv-magazine.com/2025/11/24/massive-20-gwh-sodium-ion-battery-manufacturing-plant-announced-in-china/

6. Sandia National Laboratories (2024) "Sodium-Ion Battery Development" (DOE project presentation). https://www.sandia.gov/app/uploads/sites/82/2024/08/PR2024_602_Omenya_Fred_Sodium-Batteries-1.pdf

7. IEA (2024) "Global EV Outlook 2024: Trends in electric vehicle batteries" (overview). https://www.iea.org/reports/global-ev-outlook-2024/trends-in-electric-vehicle-batteries

8. Guo et al. (2023) "Comparative life cycle assessment of sodium-ion and lithium iron phosphate batteries in the context of carbon neutrality." (conference/peer outputs summary). https://ui.adsabs.harvard.edu/abs/2023JEnSt..7208589G/abstract

(Additional sources used for context: industry press, analyst briefings; full machine-readable source list saved separately.)

---

Appendix: files produced
- /_outputs/sodium_vs_lithium_report.md (this file)
- /_outputs/sodium_vs_lithium_sources.json (machine-readable source metadata)

End of report
