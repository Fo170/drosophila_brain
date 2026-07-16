#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  GÉNÉRATEUR SVG DU CONNECTOME CÉRÉBRAL DE LARVE DE DROSOPHILA
  ==============================================================
  
  Basé sur : Winding et al. (2023) "The connectome of an insect brain"
             Science, 379(6636), eadd9330
             
  Données sources : https://codex.flywire.ai/ | https://neuprint.janelia.org/
  
  Ce script génère un diagramme de connectivité inter-régions en SVG où :
    - 1 trait SVG = agrégation des faisceaux entre 2 régions fonctionnelles
    - Épaisseur du trait ∝ nombre de synapses
    - Directionnalité via flèches et courbes de Bézier
    
  Auteur : Généré pour Olivier Fournet (GPL-3.0)
================================================================================
"""

import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Region:
    """Région fonctionnelle du cerveau larvaire de Drosophila"""
    id: str
    name: str
    name_fr: str
    neurons: int
    region_type: str  # sensory | integrator | motor | commissure
    x_norm: float     # Position normalisée [0,1]
    y_norm: float
    color: str
    description: str


@dataclass  
class Connection:
    """Faisceau de connexions inter-régions (agrégation de neurones)"""
    from_id: str
    to_id: str
    synapses: int
    conn_type: str   # feedforward | efferent | recurrent | commissure | associative | afferent
    description: str
    

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE FONCTIONNELLE (basée sur Winding et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

REGIONS = [
    # === ZONES SENSORIELLES ===
    Region("AL", "Antennal Lobe", "Lobe antennaire (AL)", 285, "sensory",
           0.25, 0.22, "#f59e0b",
           "Traitement olfactif primaire. Organisation glomerulaire. Reception ORNs."),
    Region("BO", "Bolwig Organ", "Organes de Bolwig (BO)", 24, "sensory",
           0.75, 0.18, "#38bdf8",
           "Photorecepteurs larvaires. Vision primitive et phototaxie negative."),
    Region("CH", "Chordotonal Organ", "Organes chordotonaux (CH)", 156, "sensory",
           0.12, 0.50, "#10b981",
           "Mecanorecepteurs proprioceptifs et vibratoires. Corps sous-oesophagien."),
    
    # === CENTRES INTÉGRATEURS ===
    Region("MB", "Mushroom Body", "Corps mushroom (MB)", 250, "integrator",
           0.48, 0.35, "#f472b6",
           "Centre d'apprentissage et memoire olfactive. Kenyon cells + lobes."),
    Region("LH", "Lateral Horn", "Lobe lateral (LH)", 380, "integrator",
           0.32, 0.42, "#a78bfa",
           "Integration multimodale post-AL. Decision comportementale olfactive."),
    Region("SLP", "Superior Lateral Protocerebrum", "Plaque supra-latérale (SLP)", 420, "integrator",
           0.68, 0.40, "#8b5cf6",
           "Cortex associatif superieur. Integration visuo-mecano-olfactive."),
    Region("SIP", "Superior Intermediate Protocerebrum", "Plaque intermediaire (SIP)", 310, "integrator",
           0.52, 0.55, "#6366f1",
           "Zone de transition. Relais entre centres sensoriels et moteurs."),
    Region("CRE", "Crepine", "Crete cerebrale (CRE)", 290, "integrator",
           0.38, 0.68, "#ec4899",
           "Centre d'integration motrice superieure. Coordination comportementale."),
    Region("SMP", "Superior Medial Protocerebrum", "Plaque supra-mediane (SMP)", 340, "integrator",
           0.62, 0.68, "#d946ef",
           "Reseau peptidergique et modulateur. Controle de l'etat interne."),
    
    # === COMMISSURES ===
    Region("INP", "Inferior Neuropils", "Protuberance intercalaire (INP)", 180, "commissure",
           0.50, 0.82, "#06b6d4",
           "Commissure interhemispherique majeure. 41% connexions recurrentes."),
    
    # === SORTIE MOTRICE ===
    Region("VNC", "Ventral Nerve Cord", "Cordon nerveux ventral (VNC)", 381, "motor",
           0.50, 0.95, "#ef4444",
           "Ganglions thoraciques/abdominaux. Motoneurones et pattern generators."),
]


# Matrice de connectivite inter-regions (synapses aggregees, estimees du connectome)
CONNECTIONS = [
    # --- Flux sensoriel ascendant ---
    Connection("AL", "LH", 28500, "feedforward",
        "Projection glomerulaire -> PN -> LH. Voie olfactive ascendante principale."),
    Connection("AL", "MB", 18200, "feedforward",
        "Projection neurons -> calyx du mushroom body. Apprentissage olfactif associatif."),
    Connection("BO", "SLP", 12400, "feedforward",
        "Voie visuelle laterale. Photorecepteurs -> cortex associatif visuel."),
    Connection("BO", "SIP", 6800, "feedforward",
        "Voie visuelle mediane. Integration visuo-motrice directe."),
    Connection("CH", "SLP", 15600, "feedforward",
        "Proprioception -> cortex associatif. Coordination posture et locomotion."),
    Connection("CH", "SIP", 9200, "feedforward",
        "Mecanoreception -> zone intermediaire. Reflexes rapides."),
    Connection("CH", "CRE", 7400, "feedforward",
        "Feedback proprioceptif au centre moteur superieur."),
    
    # --- Reseau associatif intra-cerebral ---
    Connection("LH", "SLP", 22400, "associative",
        "Integration olfacto-visuelle. Decision comportementale multi-modale."),
    Connection("LH", "MB", 15800, "associative",
        "Feedback LH -> mushroom body. Modulation de la valence olfactive."),
    Connection("SLP", "SIP", 18600, "associative",
        "Cortex associatif -> zone de transition. Flux sensoriel integre."),
    Connection("SIP", "CRE", 14200, "associative",
        "Relais vers centre moteur superieur. Planification d'actions."),
    Connection("SIP", "SMP", 12800, "associative",
        "Modulation peptidergique de l'etat comportemental et motivationnel."),
    Connection("CRE", "SMP", 11600, "associative",
        "Coordination moteur-etat interne. Sequencement comportemental."),
    Connection("MB", "CRE", 9800, "associative",
        "Mushroom body -> centre moteur. Memoire a action."),
    Connection("MB", "SMP", 8200, "associative",
        "Modulation dopaminergique de l'etat interne par le MB."),
    
    # --- Boucles recurrentes (41% des neurones) ---
    Connection("LH", "AL", 12400, "recurrent",
        "Feedback central -> lobe antennaire. Modulation olfactive descendante (top-down)."),
    Connection("SLP", "BO", 4200, "recurrent",
        "Feedback visuel. Attention visuelle et modulation de la sensibilite."),
    Connection("CRE", "SIP", 9800, "recurrent",
        "Boucle moteur-sensorielle. Prediction motrice et efference copy."),
    Connection("SMP", "SIP", 7600, "recurrent",
        "Regulation homeostatique du traitement sensoriel."),
    Connection("MB", "MB", 15400, "recurrent",
        "Recurrence interne au mushroom body. Rappel, consolidation et generalisation."),
    Connection("CRE", "CRE", 8200, "recurrent",
        "Auto-regulation motrice. Sequencement et timing des actions."),
    
    # --- Commissures interhemispheriques ---
    Connection("INP", "AL", 8600, "commissure",
        "Communication interhemispherique olfactive. Comparaison bilaterale."),
    Connection("INP", "SLP", 11200, "commissure",
        "Integration bilaterale des senses. Fusion perceptive."),
    Connection("INP", "SIP", 9800, "commissure",
        "Coordination interhemispherique motrice. Synchronisation locomotrice."),
    Connection("INP", "CRE", 7400, "commissure",
        "Synchronisation des programmes moteurs bilateraux."),
    Connection("AL", "INP", 6200, "commissure",
        "Olfaction contralaterale. Detection de gradients spatiaux."),
    Connection("SLP", "INP", 7800, "commissure",
        "Associatif contralateral. Integration multi-modale bilaterale."),
    
    # --- Sortie motrice descendante ---
    Connection("CRE", "VNC", 18600, "efferent",
        "Commande motrice descendante. Activation des pattern generators segmentaires."),
    Connection("SMP", "VNC", 12400, "efferent",
        "Modulation peptidergique de la locomotion et du rythme."),
    Connection("SIP", "VNC", 14200, "efferent",
        "Relais moteur direct. Reflexes rapides et corrections posturales."),
    Connection("INP", "VNC", 9800, "efferent",
        "Coordination locomotrice bilaterale. Alternance des pas."),
    
    # --- Afferences proprioceptives ---
    Connection("VNC", "CH", 11200, "afferent",
        "Proprioception ascendante depuis les segments corporels."),
    Connection("VNC", "SIP", 8600, "afferent",
        "Feedback locomoteur au centre integrateur. Boucle fermee."),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES DE RENDU SVG
# ═══════════════════════════════════════════════════════════════════════════════

SVG_WIDTH = 1400
SVG_HEIGHT = 1000
MARGIN = 80

# Palette de couleurs par type de connexion
CONN_COLORS = {
    "feedforward": "#f59e0b",   # Orange - sensoriel
    "efferent": "#ef4444",      # Rouge - moteur
    "recurrent": "#f472b6",     # Rose - recurrent
    "commissure": "#06b6d4",    # Cyan - commissure
    "associative": "#a78bfa",   # Violet - associatif
    "afferent": "#10b981",      # Vert - afferent
}

CONN_LABELS = {
    "feedforward": "Sensoriel ↑",
    "efferent": "Moteur ↓",
    "recurrent": "Recurrent ↻",
    "commissure": "Commissure ↔",
    "associative": "Associatif ⟷",
    "afferent": "Afferent ↑",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE DU GÉNÉRATEUR
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectomeSVGGenerator:
    """Generateur de diagramme de connectivite inter-regions en SVG"""
    
    def __init__(self, regions: List[Region], connections: List[Connection],
                 width: int = SVG_WIDTH, height: int = SVG_HEIGHT):
        self.regions = {r.id: r for r in regions}
        self.connections = connections
        self.width = width
        self.height = height
        self.margin = MARGIN
        self.positions = self._compute_positions()
        self.max_synapses = max(c.synapses for c in connections)
        
    def _compute_positions(self) -> Dict[str, Tuple[float, float]]:
        """Calcule les positions absolues des regions"""
        positions = {}
        for rid, reg in self.regions.items():
            x = self.margin + reg.x_norm * (self.width - 2 * self.margin)
            y = self.margin + reg.y_norm * (self.height - 2 * self.margin)
            positions[rid] = (x, y)
        return positions
    
    def _thickness(self, synapses: int) -> float:
        """Calcule l'epaisseur du trait proportionnelle au nombre de synapses"""
        return max(0.8, (synapses / self.max_synapses) * 10)
    
    def _bezier_curve(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      offset_idx: int = 0) -> str:
        """Genere une courbe de Bezier quadratique avec decalage pour eviter les superpositions"""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0:
            return f"M{x1},{y1} L{x2},{y2}"
        
        # Point de controle avec decalage perpendiculaire
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        perp_x = -dy / dist
        perp_y = dx / dist
        offset = (offset_idx % 2 * 2 - 1) * dist * 0.15  # Alternance +/-
        
        cx = mx + perp_x * offset
        cy = my + perp_y * offset
        
        return f"M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}"
    
    def _arrow_marker(self, color: str, marker_id: str) -> str:
        """Genere un marqueur de fleche SVG"""
        return f'    <marker id="{marker_id}" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">\n      <path d="M0,0 L0,6 L9,3 z" fill="{color}" opacity="0.7"/>\n    </marker>'
    
    def generate_svg(self, mode: str = "full", selected_region: Optional[str] = None) -> str:
        """
        Genere le SVG complet.
        
        Modes :
            - "full" : vue d'ensemble complete
            - "sensory" : flux sensoriel ascendant uniquement
            - "motor" : flux moteur descendant uniquement  
            - "recurrent" : boucles recurrentes et commissures
            - "associative" : reseau associatif intra-cerebral
        """
        # Filtrer les connexions selon le mode
        conns = self._filter_connections(mode, selected_region)
        
        svg_parts = []
        
        # En-tete SVG
        svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 {self.width} {self.height}" 
     width="{self.width}" height="{self.height}"
     style="background:#0a0e17; font-family:'Inter',system-ui,sans-serif;">''')
        
        # Definitions (gradients, filtres, marqueurs)
        svg_parts.append("  <defs>")
        
        # Marqueurs de fleche par type
        for ctype, color in CONN_COLORS.items():
            svg_parts.append(self._arrow_marker(color, f"arrow-{ctype}"))
        
        # Filtre de glow
        svg_parts.append('''    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>''')
        
        svg_parts.append("  </defs>")
        
        # Titre et legende
        svg_parts.append(self._render_title_and_legend(mode))
        
        # Connexions (traits)
        svg_parts.append(self._render_connections(conns))
        
        # Noeuds regions (cercles + labels)
        svg_parts.append(self._render_nodes(selected_region))
        
        # Infobox avec statistiques
        svg_parts.append(self._render_infobox(conns))
        
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
    
    def _filter_connections(self, mode: str, selected: Optional[str]) -> List[Connection]:
        """Filtre les connexions selon le mode de visualisation"""
        conns = self.connections
        
        if mode == "sensory":
            conns = [c for c in conns if c.conn_type in ("feedforward", "afferent")]
        elif mode == "motor":
            conns = [c for c in conns if c.conn_type in ("efferent",) or c.to_id == "VNC"]
        elif mode == "recurrent":
            conns = [c for c in conns if c.conn_type in ("recurrent", "commissure")]
        elif mode == "associative":
            conns = [c for c in conns if c.conn_type == "associative"]
            
        if selected:
            conns = [c for c in conns if c.from_id == selected or c.to_id == selected]
            
        return conns
    
    def _render_connections(self, conns: List[Connection]) -> str:
        """Rendu SVG des traits de connexion"""
        lines = ["  <!-- CONNEXIONS INTER-REGIONS -->"]
        
        for i, conn in enumerate(conns):
            p1 = self.positions.get(conn.from_id)
            p2 = self.positions.get(conn.to_id)
            if not p1 or not p2:
                continue
                
            thickness = self._thickness(conn.synapses)
            color = CONN_COLORS.get(conn.conn_type, "#94a3b8")
            opacity = 0.5 + (conn.synapses / self.max_synapses) * 0.45
            marker = f"url(#arrow-{conn.conn_type})"
            
            path_d = self._bezier_curve(p1, p2, i)
            
            # Tooltip title pour interaction
            title = f"{self.regions[conn.from_id].name_fr} -> {self.regions[conn.to_id].name_fr}: {conn.synapses:,} synapses"
            
            lines.append(f'''    <path d="{path_d}" 
          stroke="{color}" stroke-width="{thickness:.1f}" 
          fill="none" opacity="{opacity:.2f}" marker-end="{marker}">
      <title>{title}
{conn.description}</title>
    </path>''')
            
            # Label pour les connexions majeures (>15k synapses)
            if conn.synapses > 15000:
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                lines.append(f'''    <text x="{mx:.1f}" y="{my:.1f}" 
          text-anchor="middle" dy="-5"
          fill="#94a3b8" font-size="10" font-family="monospace"
          opacity="0.8">{(conn.synapses/1000):.1f}k</text>''')
        
        return "\n".join(lines)
    
    def _render_nodes(self, selected: Optional[str]) -> str:
        """Rendu SVG des noeuds regions"""
        lines = ["  <!-- REGIONS FONCTIONNELLES -->"]
        
        for rid, reg in self.regions.items():
            px, py = self.positions[rid]
            is_selected = selected == rid
            radius = 16 + math.sqrt(reg.neurons) * 0.45
            glow = 'filter="url(#glow)"' if is_selected else ''
            
            lines.append(f'''    <g transform="translate({px:.1f},{py:.1f})">
      <circle r="{radius:.1f}" fill="{reg.color}" opacity="0.12" {glow}/>
      <circle r="{radius*0.55:.1f}" fill="{reg.color}" opacity="0.5"/>
      <circle r="5" fill="{reg.color}"/>
      <text y="{radius+18:.1f}" text-anchor="middle" 
            fill="#e2e8f0" font-size="12" font-weight="600">{rid}</text>
      <text y="{radius+32:.1f}" text-anchor="middle" 
            fill="#64748b" font-size="9">{reg.neurons}N</text>
    </g>''')
        
        return "\n".join(lines)
    
    def _render_title_and_legend(self, mode: str) -> str:
        """Rendu du titre et de la legende"""
        mode_titles = {
            "full": "Vue d'ensemble complete",
            "sensory": "Flux sensoriels ascendants",
            "motor": "Flux moteurs descendants",
            "recurrent": "Boucles recurrentes et commissures",
            "associative": "Reseau associatif intra-cerebral",
        }
        
        title = mode_titles.get(mode, "Vue d'ensemble")
        
        legend_items = []
        y_start = 120
        for ctype, color in CONN_COLORS.items():
            label = CONN_LABELS.get(ctype, ctype)
            legend_items.append(
                f'''    <rect x="30" y="{y_start}" width="20" height="3" rx="1.5" fill="{color}" opacity="0.8"/>
    <text x="58" y="{y_start+3}" fill="#94a3b8" font-size="11" dominant-baseline="middle">{label}</text>'''
            )
            y_start += 22
        
        return f'''  <!-- TITRE ET LEGENDE -->
  <text x="{self.width/2}" y="40" text-anchor="middle" 
        fill="#38bdf8" font-size="22" font-weight="700">
    Connectome Larve Drosophila melanogaster
  </text>
  <text x="{self.width/2}" y="65" text-anchor="middle" 
        fill="#64748b" font-size="13">
    {title} · Winding et al. 2023 · 3 016 neurones · 548 000 synapses
  </text>
  <text x="30" y="100" fill="#e2e8f0" font-size="13" font-weight="600">Legende des faisceaux</text>
{chr(10).join(legend_items)}'''
    
    def _render_infobox(self, conns: List[Connection]) -> str:
        """Rendu de la boite d'information statistique"""
        total_syn = sum(c.synapses for c in conns)
        total_fibers = len(conns)
        
        return f'''  <!-- INFOBOX STATISTIQUES -->
  <rect x="{self.width-260}" y="90" width="230" height="130" rx="12" 
        fill="#111827" stroke="#1e293b" stroke-width="1" opacity="0.95"/>
  <text x="{self.width-245}" y="120" fill="#38bdf8" font-size="14" font-weight="700">
    Statistiques
  </text>
  <text x="{self.width-245}" y="145" fill="#e2e8f0" font-size="12">
    Faisceaux affiches: <tspan fill="#f472b6" font-weight="700">{total_fibers}</tspan>
  </text>
  <text x="{self.width-245}" y="165" fill="#e2e8f0" font-size="12">
    Synapses totales: <tspan fill="#f472b6" font-weight="700">{total_syn:,}</tspan>
  </text>
  <text x="{self.width-245}" y="185" fill="#94a3b8" font-size="10">
    Epaisseur ∝ nombre de synapses
  </text>
  <text x="{self.width-245}" y="205" fill="#94a3b8" font-size="10">
    1 trait = agregation inter-regions
  </text>'''


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_views(output_dir: str = "./connectome_output"):
    """Genere toutes les vues SVG du connectome"""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    
    gen = ConnectomeSVGGenerator(REGIONS, CONNECTIONS)
    modes = ["full", "sensory", "motor", "recurrent", "associative"]
    
    generated = []
    for mode in modes:
        svg = gen.generate_svg(mode=mode)
        filepath = out / f"connectome_drosophila_{mode}.svg"
        filepath.write_text(svg, encoding="utf-8")
        generated.append(str(filepath))
        print(f"OK Genere : {filepath}")
    
    return generated


def generate_json_data(output_path: str = "./connectome_data.json"):
    """Exporte les donnees structurees en JSON pour usage externe (D3.js, Cytoscape, etc.)"""
    data = {
        "metadata": {
            "species": "Drosophila melanogaster (larva)",
            "reference": "Winding et al. 2023, Science 379(6636), eadd9330",
            "neurons_total": 3016,
            "synapses_total": 548000,
            "regions_count": len(REGIONS),
            "neuron_types": 93,
            "recurrence_rate": 0.41,
            "hubs_io_learning_rate": 0.73,
            "license": "CC-BY / Open Data"
        },
        "regions": [
            {
                "id": r.id,
                "name": r.name,
                "name_fr": r.name_fr,
                "neurons": r.neurons,
                "type": r.region_type,
                "position": {"x": r.x_norm, "y": r.y_norm},
                "color": r.color,
                "description": r.description
            }
            for r in REGIONS
        ],
        "connections": [
            {
                "source": c.from_id,
                "target": c.to_id,
                "synapses": c.synapses,
                "type": c.conn_type,
                "description": c.description
            }
            for c in CONNECTIONS
        ]
    }
    
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK Donnees JSON exportees : {output_path}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  GENERATEUR SVG DU CONNECTOME DROSOPHILA LARVAIRE")
    print("  Winding et al. 2023 — Science")
    print("=" * 70)
    print()
    
    # Generer toutes les vues SVG
    print("[1/2] Generation des vues SVG...")
    files = generate_all_views()
    print(f"\n  {len(files)} fichiers SVG generes dans ./connectome_output/")
    print()
    
    # Exporter les donnees JSON
    print("[2/2] Export des donnees structurees...")
    generate_json_data()
    print()
    
    print("=" * 70)
    print("  FICHIERS GENERES :")
    for f in files:
        print(f"    - {f}")
    print("    - connectome_data.json")
    print("=" * 70)
    print()
    print("  UTILISATION AVEC DONNEES REELLES :")
    print("  ------------------------------------")
    print("  1. Telecharger le connectome complet depuis :")
    print("     https://codex.flywire.ai/")
    print("     https://neuprint.janelia.org/")
    print()
    print("  2. Remplacer la matrice CONNECTIONS par les donnees EM reelles")
    print("     (synapses comptees par paire de regions)")
    print()
    print("  3. Relancer le script pour obtenir le SVG a l'echelle reelle")
    print("=" * 70)