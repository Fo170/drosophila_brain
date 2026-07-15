"""Monde 3D avec affichage graphique temps reel + texte terminal"""
import sys, numpy as np
sys.path.insert(0, '.')
from config import DT
from core.network import BrainNetwork
from learning.reinforcement_learning import DANModulatedLearning
from world.world_3d import VirtualWorld3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class World3DVisualizer:
    def __init__(self, world_size=(50,50,20), seed=42):
        print("="*70)
        print("  MONDE 3D - Drosophila melanogaster")
        print("  Fenetre graphique + Informations terminal")
        print("="*70)
        self.world = VirtualWorld3D(size=world_size, seed=seed)
        self.network = BrainNetwork(seed=seed)
        self.learner = DANModulatedLearning(self.network)
        self.fig = plt.figure(figsize=(16,10))
        self.fig.patch.set_facecolor('#0d1117')
        self.ax_world = self.fig.add_subplot(2,2,1,projection='3d')
        self.ax_world.set_facecolor('#161b22')
        self.ax_world.set_title('MONDE 3D - Trajectoire', color='white', fontsize=12, fontweight='bold')
        for ax_label in ['x','y','z']: getattr(self.ax_world, f'set_{ax_label}label')(ax_label.upper(), color='white')
        self.ax_world.tick_params(colors='white')
        self.ax_world.set_xlim(0,world_size[0]); self.ax_world.set_ylim(0,world_size[1]); self.ax_world.set_zlim(0,world_size[2])
        self.ax_brain = self.fig.add_subplot(2,2,2)
        self.ax_brain.set_facecolor('#161b22')
        self.ax_brain.set_title('ACTIVITE CEREBRALE', color='white', fontsize=12, fontweight='bold')
        self.ax_brain.set_xlabel('Temps (ms)', color='white'); self.ax_brain.set_ylabel('Activite', color='white')
        self.ax_brain.tick_params(colors='white'); self.ax_brain.set_ylim(0,1)
        self.ax_info = self.fig.add_subplot(2,2,3)
        self.ax_info.set_facecolor('#161b22'); self.ax_info.axis('off')
        self.ax_stimuli = self.fig.add_subplot(2,2,4)
        self.ax_stimuli.set_facecolor('#161b22')
        self.ax_stimuli.set_title('STIMULI SENSORIELS', color='white', fontsize=12, fontweight='bold')
        self.ax_stimuli.tick_params(colors='white')
        plt.tight_layout(); plt.subplots_adjust(top=0.93,hspace=0.3,wspace=0.3)
        self.time_history=[]; self.brain_history={k:[] for k in ['sensory','KC','MBON','DAN','DN_VNC']}
        self.colors={'sensory':'#ff6b6b','KC':'#48dbfb','MBON':'#ff9ff3','DAN':'#54a0ff','DN_VNC':'#1dd1a1',
                     'olfactory':'#2ecc71','gustatory':'#f1c40f','thermal':'#e74c3c','visual':'#3498db'}
        print("Initialisation terminee!\nFermez la fenetre pour arreter\n")

    def _draw_world(self):
        self.ax_world.clear(); self.ax_world.set_facecolor('#161b22')
        self.ax_world.set_title('MONDE 3D - Trajectoire', color='white', fontsize=11, fontweight='bold')
        for ax_label in ['x','y','z']: getattr(self.ax_world, f'set_{ax_label}label')(ax_label.upper(), color='white')
        self.ax_world.tick_params(colors='white')
        if len(self.world.trajectory)>1:
            traj=np.array(self.world.trajectory)
            self.ax_world.plot(traj[:,0],traj[:,1],traj[:,2],'c-',alpha=0.5,linewidth=1,label='Trajectoire')
        pos=self.world.insect_pos
        self.ax_world.scatter(*pos,c='cyan',s=100,marker='o',edgecolors='white',linewidths=2,label='Larve')
        orient=self.world.insect_orientation
        self.ax_world.quiver(pos[0],pos[1],pos[2],orient[0]*3,orient[1]*3,orient[2]*3,color='yellow',arrow_length_ratio=0.3,linewidth=2)
        for odor in self.world.odor_sources:
            color='lime' if odor['type']=='attractive' else 'orange' if odor['type']=='aversive' else 'gray'
            self.ax_world.scatter(*odor['pos'],c=color,s=80,marker='*',alpha=0.7)
        for food in self.world.food_sources:
            color='gold' if not food['consumed'] else 'gray'
            self.ax_world.scatter(*food['pos'],c=color,s=60,marker='o',alpha=0.8)
        for threat in self.world.threat_zones:
            self.ax_world.scatter(*threat['pos'],c='red',s=60,marker='X',alpha=0.7)
        self.ax_world.legend(loc='upper left',facecolor='#161b22',edgecolor='white',labelcolor='white',fontsize=8)

    def _draw_brain_activity(self,current_time):
        self.ax_brain.clear(); self.ax_brain.set_facecolor('#161b22')
        self.ax_brain.set_title('ACTIVITE CEREBRALE',color='white',fontsize=11,fontweight='bold')
        self.ax_brain.set_xlabel('Temps (ms)',color='white'); self.ax_brain.set_ylabel('Activite moyenne',color='white')
        self.ax_brain.tick_params(colors='white'); self.ax_brain.set_ylim(0,1)
        self.ax_brain.grid(True,alpha=0.2,color='gray')
        region_act=self.network.get_region_activity(); self.time_history.append(current_time)
        for key in self.brain_history: self.brain_history[key].append(region_act.get(key,0))
        max_hist=500
        if len(self.time_history)>max_hist:
            self.time_history=self.time_history[-max_hist:]
            for key in self.brain_history: self.brain_history[key]=self.brain_history[key][-max_hist:]
        for key,values in self.brain_history.items():
            if len(values)>1: self.ax_brain.plot(self.time_history,values,color=self.colors.get(key,'white'),label=key,linewidth=1.5)
        self.ax_brain.legend(loc='upper right',facecolor='#161b22',edgecolor='white',labelcolor='white',fontsize=8)
        self.ax_brain.set_xlim(max(0,current_time-500),current_time+50)

    def _draw_stimuli(self,stimuli):
        self.ax_stimuli.clear(); self.ax_stimuli.set_facecolor('#161b22')
        self.ax_stimuli.set_title('STIMULI ACTUELS',color='white',fontsize=11,fontweight='bold')
        self.ax_stimuli.tick_params(colors='white')
        labels=['Olfaction','Gustation','Thermo','Visuel']
        values=[stimuli['olfactory']['total'],stimuli['gustatory'],stimuli['thermal'],stimuli['visual']]
        colors_list=[self.colors['olfactory'],self.colors['gustatory'],self.colors['thermal'],self.colors['visual']]
        bars=self.ax_stimuli.bar(labels,values,color=colors_list,alpha=0.8,edgecolor='white')
        self.ax_stimuli.set_ylim(0,1.2); self.ax_stimuli.set_ylabel('Intensite',color='white')
        for bar,val in zip(bars,values):
            height=bar.get_height()
            self.ax_stimuli.text(bar.get_x()+bar.get_width()/2.,height,f'{val:.2f}',ha='center',va='bottom',color='white',fontsize=9)

    def _draw_info(self,current_time,events):
        self.ax_info.clear(); self.ax_info.set_facecolor('#161b22'); self.ax_info.axis('off')
        state=self.world.get_state(); region_act=self.network.get_region_activity()
        info_str=f"""SIMULATION TEMPS REEL - Drosophila 3D
Temps:        {current_time:>8.1f} ms
Position:     ({state['insect_pos'][0]:>5.1f}, {state['insect_pos'][1]:>5.1f}, {state['insect_pos'][2]:>5.1f})
Vitesse:      {self.world.insect_speed:>8.3f}

ACTIVITE CEREBRALE:
  Sensoriel:  {region_act['sensory']:>8.4f}    KC:      {region_act['KC']:>8.4f}
  MBON:       {region_act['MBON']:>8.4f}    DAN:     {region_act['DAN']:>8.4f}
  DN_VNC:     {region_act['DN_VNC']:>8.4f}    CN:      {region_act['CN']:>8.4f}

STATISTIQUES:
  Distance:   {state['distance_traveled']:>8.1f}    Nourriture: {state['food_consumed']:>3d}
  Menaces:    {state['threats_encountered']:>8d}    Interactions: {state['n_interactions']:>3d}

DERNIER EVENEMENT:
  {events.get('last_event','Aucun')}"""
        self.ax_info.text(0.02,0.98,info_str,transform=self.ax_info.transAxes,color='#00ff88',fontsize=9,verticalalignment='top',fontfamily='monospace',linespacing=1.1)

    def run(self,duration_ms=5000.0,update_interval_ms=100):
        n_steps=int(duration_ms/DT); update_every=int(update_interval_ms/DT)
        print(f"\nDemarrage simulation: {duration_ms}ms")
        print(f"Mise a jour: tous les {update_interval_ms}ms")
        print("Fermez la fenetre pour arreter\n")
        plt.ion(); plt.show(block=False)
        last_event="Aucun"
        for step in range(n_steps):
            current_time=step*DT
            stimuli=self.world.get_sensory_input_3d()
            if stimuli['olfactory']['total']>0.1: self.network.apply_stimulus('olfactory',stimuli['olfactory']['total'],10)
            if stimuli['gustatory']>0.1: self.network.apply_stimulus('gustatory',stimuli['gustatory'],10)
            if stimuli['thermal']>0.1: self.network.apply_stimulus('thermal',stimuli['thermal'],10)
            if stimuli['visual']>0.3: self.network.apply_stimulus('visual',stimuli['visual'],10)
            if stimuli['mechanosensory']>0.1: self.network.apply_stimulus('mechano',stimuli['mechanosensory'],10)
            self.network.step(DT)
            dn_vnc=self.network.dn_neurons['VNC']
            if len(dn_vnc)>0:
                left_act=np.mean([n.output for n in dn_vnc[:len(dn_vnc)//2]])
                right_act=np.mean([n.output for n in dn_vnc[len(dn_vnc)//2:]])
                speed=(left_act+right_act)*1.5; turn=(right_act-left_act)*np.pi*0.5
            else: speed=0.3+np.random.normal(0,0.1); turn=np.random.normal(0,0.3)
            self.world.move_insect_3d(speed,turn,0.0)
            events=self.world.check_interactions_3d()
            if events['reward']>0:
                self.learner.apply_reward(events['reward'])
                last_event=f"NOURRITURE! Recompense={events['reward']:.2f}"
                print(f"  [t={current_time:.0f}ms] {last_event}")
            if events['punishment']>0:
                self.learner.apply_punishment(events['punishment'])
                last_event=f"DANGER! Punition={events['punishment']:.2f}"
                print(f"  [t={current_time:.0f}ms] {last_event}")
            if step%update_every==0:
                self._draw_world(); self._draw_brain_activity(current_time)
                self._draw_stimuli(stimuli); self._draw_info(current_time,{'last_event':last_event})
                self.fig.canvas.draw(); self.fig.canvas.flush_events()
                last_event="Aucun"
            if not plt.fignum_exists(self.fig.number):
                print("\nFenetre fermee - Arret"); break
        plt.ioff()
        print("\n"+"="*70); print("SIMULATION TERMINEE"); print("="*70)
        final_state=self.world.get_state()
        print(f"\nResultats:")
        print(f"  Distance: {final_state['distance_traveled']:.1f}")
        print(f"  Nourriture: {final_state['food_consumed']}")
        print(f"  Menaces: {final_state['threats_encountered']}")
        learning=self.learner.get_learning_summary()
        print(f"\nApprentissage:")
        print(f"  Signaux DAN: {learning['n_dan_signals']}")
        print(f"  DAN moyen: {learning['mean_dan']:.4f}")
        print("\nFermez la fenetre pour quitter")
        plt.show()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description='Monde 3D temps reel')
    parser.add_argument('--duration',type=float,default=10000.0,help='Duree ms')
    parser.add_argument('--update',type=float,default=100.0,help='Intervalle ms')
    args=parser.parse_args()
    viz=World3DVisualizer()
    viz.run(duration_ms=args.duration,update_interval_ms=args.update)
