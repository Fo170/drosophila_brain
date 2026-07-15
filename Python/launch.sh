#!/bin/bash
# Script de lancement du simulateur Drosophila

echo "=========================================="
echo "  Simulateur Cerveau Drosophila"
echo "=========================================="
echo ""

# Vérifier Python
python3 --version || { echo "Python 3 requis"; exit 1; }

# Options
if [ "$1" == "--test" ]; then
    echo "▶ Mode test et validation"
    python3 test_validation.py
elif [ "$1" == "--advanced" ]; then
    echo "▶ Mode avancé (circuits + monde 3D)"
    python3 main_advanced.py
elif [ "$1" == "--interactive" ]; then
    echo "▶ Mode interactif"
    python3 main.py --interactive
elif [ "$1" == "--visual" ]; then
    echo "▶ Mode visualisation temps réel"
    python3 main.py --visual
elif [ "$1" == "--world3d" ]; then
    echo "▶ Mode monde 3D (insecte en mouvement)"
    shift
    python3 world_3d_demo.py "$@"
else
    echo "Usage: ./launch.sh [OPTION] [ARGS]"
    echo ""
    echo "Options:"
    echo "  --test        Tests et validation (7 tests)"
    echo "  --advanced    Simulation avancée complète"
    echo "  --interactive Mode interactif"
    echo "  --visual      Visualisation temps réel 2D"
    echo "  --world3d     Monde 3D avec insecte en mouvement"
    echo ""
    echo "Exemples monde 3D:"
    echo "  ./launch.sh --world3d                    # Démo 5s"
    echo "  ./launch.sh --world3d --duration 10000   # 10 secondes"
    echo "  ./launch.sh --world3d --save trajectoire.png"
    echo ""
    echo "Par défaut: démonstration simple"
    python3 main.py
fi
