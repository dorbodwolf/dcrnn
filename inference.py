from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        print('loading model at checkpoint ', args.epoch)
        supervisor.load_model_inference(args.log_dir, args.epoch)

        if args.mode == 'sample':
            supervisor.test()
        elif args.mode == 'map':
            supervisor.mapping(args.metainfo, args.index_file, args.ref_tif, args.write_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--log_dir', default=None, type=str, help='dir to load weights')
    parser.add_argument('--epoch', default=0, type=int, help='Set to true to only use cpu.')
    parser.add_argument('--mode', default=None, type=str, help='选择推理模式，可以是sample：抽样一个像素点可视化与真值的时序曲线拟合效果；map：绘制所有像素的预测结果到地图')
    parser.add_argument('--metainfo', default='/Users/jade_mayer/projects/geospatial/冰川流速预测/code/data/KYAGR/kyagar_valid_data.csv', type=str,
                         help='数据集构造的csv，包含id和datatime列，其中id列的顺序和测试集的数据顺序一致；测试集对应日期是datatime尾部与测试集长度对应的的切片')
    parser.add_argument('--index_file',default='/Users/jade_mayer/projects/geospatial/冰川流速预测/code/data/KYAGR/data_pnts_rowscols_index.csv', type=str, help='id和原始tif行列号的映射关系，帮助还原id对应的位置')
    parser.add_argument('--ref_tif', default='/Users/jade_mayer/projects/geospatial/冰川流速预测/韩师兄原始数据/冰川流速数据集201410_202107/kyagar201410/dbland_cropped.tif', type=str, help='参考tif，用于还原坐标系统')
    parser.add_argument('--write_dir', default='/Users/jade_mayer/projects/geospatial/冰川流速预测/韩师兄原始数据/冰川流速数据集201410_202107/输出预测', type=str, help='输出预测tif路径')
    args = parser.parse_args()
    main(args)
