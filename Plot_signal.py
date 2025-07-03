import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Plot_signal:
    def __init__(self,file, sheet_name, column_name, file_labels,output_file):
        df = pd.read_excel(file, sheet_name = sheet_name)
        self.signal = df[column_name].values
        self.times = df.index.values
        lf = pd.read_excel(file_labels)
        self.labels = lf
        self.output_file = output_file
        
    #=== Plot the raw graph ====
    def _init_plot(self):
       self.fig, self.ax = plt.subplots(figsize = (15,4))
       self.ax.plot(self.signal, label = "Signal", color = "Blue")
       self.ax.set_xlabel("Frame")
       self.ax.set_ylabel("Signal")
       self.ax.grid(True)
       
    #===prepare a dictionary with the labels which define the beggining and last of each frame
    def _group_labels(self):
        self.labels['new_segment'] = (self.labels['Labels'] != self.labels['Labels'].shift(1)).cumsum()
        segments_df = (self.labels.groupby('new_segment').agg(label=('Labels','first'), start=('Frame','min'), end=('Frame','max')).reset_index(drop=True))
        seg_groups = segments_df.groupby('label').groups #gives index
        #group in dictionary
        label_segments = {int(lbl) : segments_df.loc[idxs, ['start', 'end']].values.tolist() for lbl, idxs in seg_groups.items()}
        return segments_df , label_segments
    
    #===plot the events which were marked================
    def _plot_labels(self, segments_df):
        cmap1 = ['white','red', 'green','blue','magenta','cyan','yellow','brown']
        seen = set()
        for _, row in segments_df.iterrows():  
            lbl, start, end = row['label'], row['start'], row['end']
            color = cmap1[lbl]
            if lbl not in seen:
               legend_label = f'{lbl}' 
            else:
                legend_label = '_nolegend_'
                
            seen.add(lbl)
            self.ax.axvspan(start, end, color = color, alpha=0.3, label=legend_label)
        self.ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
  
        
    #===plot graph with labels and save============
    def _mark_labels(self):
        segments_df, label_segments = self._group_labels()
        segments_df.to_excel(self.output_file + "labels_location.xlsx", index = False)
        self._init_plot()
        self._plot_labels(segments_df)
        plt.tight_layout()
        plt.show()
        