import vame

class vame_project:
    def __init__(self, working_dir, project_name, csv_path):
        self.working_dir = working_dir
        self.project_name = project_name
        self.csv_path = csv_path
        
    def configuration(self):
        config = vame.init_new_project(project_name = self.project_name, poses_estimations = [self.csv_path],source_software   = "DeepLabCut",videos = [], working_directory = str(self.working_dir), video_type= None)
        return config
    
    