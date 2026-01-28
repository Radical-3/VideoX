from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.prj_dir = r'd:\code\VideoX\SeqTrack'
    settings.save_dir = r'd:\code\VideoX\SeqTrack'
    settings.davis_dir = ''
    settings.got10k_path = r'd:\code\VideoX\SeqTrack\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = r'd:\code\VideoX\SeqTrack\lib\test\networks'
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = r'd:\code\VideoX\SeqTrack\lib\test\result_plots'
    settings.results_path = r'd:\code\VideoX\SeqTrack\lib\test\tracking_results'
    settings.segmentation_path = r'd:\code\VideoX\SeqTrack\lib\test\segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
