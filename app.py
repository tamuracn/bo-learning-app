import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib; matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
import threading
import uuid
import numpy as np

app = Flask(__name__)

_jobs = {}
_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/landscape')
def landscape():
    from toy_model.toy_function import toy_function
    layer = int(request.args.get('layer', 1))
    sigma = float(request.args.get('sigma', 0.5))
    grid  = int(request.args.get('grid', 50))
    show_donor = request.args.get('show_donor', 'false') == 'true'

    g = np.linspace(0, 3, grid)
    xx, yy = np.meshgrid(g, g)
    zz = np.vectorize(lambda x, y: toy_function(x, y, layer, sigma=sigma, noise_scale=0.0))(xx, yy)

    result = {'x': g.tolist(), 'y': g.tolist(), 'z': zz.tolist()}

    if show_donor:
        donor_layer = int(request.args.get('donor_layer', 1))
        z_donor = np.vectorize(lambda x, y: toy_function(x, y, donor_layer, sigma=sigma, noise_scale=0.0))(xx, yy)
        result['z_donor'] = z_donor.tolist()

    return jsonify(result)


@app.route('/api/landscape3d')
def landscape3d():
    from toy_model.toy_function import toy_function
    sigma = float(request.args.get('sigma', 0.5))
    grid  = int(request.args.get('grid', 35))
    donor  = int(request.args.get('donor', 1))
    target = int(request.args.get('target', 2))

    g = np.linspace(0, 3, grid)
    xx, yy = np.meshgrid(g, g)
    layers = []
    for n in sorted(set([donor, target])):
        zz = np.array([[toy_function(float(xx[i, j]), float(yy[i, j]), n,
                                     sigma=sigma, noise_scale=0.0)
                        for j in range(grid)] for i in range(grid)])
        layers.append({'n': n, 'z': zz.tolist()})

    return jsonify({'x': g.tolist(), 'y': g.tolist(), 'layers': layers, 'donor': donor, 'target': target})


@app.route('/api/run', methods=['POST'])
def run():
    from bo_runner import run_experiment
    config = request.json
    job_id = str(uuid.uuid4())[:8]

    job = {
        'status': 'running',
        'config': config,
        'events': [],
        'gp_maps': {},
        'done': False,
        'error': None,
        'current': {'method': None, 'seed': 0, 'iter': 0}
    }
    with _lock:
        _jobs[job_id] = job

    def on_event(event):
        with _lock:
            j = _jobs[job_id]
            gp_map = event.pop('gp_map', None)
            j['events'].append(dict(event))
            j['current'] = {
                'method': event['method'],
                'seed': event['seed'],
                'iter': event['iter']
            }
            if gp_map:
                key = f"{event['method']}_{event['seed']}_{event['iter']}"
                j['gp_maps'][key] = gp_map

    def worker():
        try:
            run_experiment(config, on_event)
        except Exception:
            import traceback
            with _lock:
                _jobs[job_id]['error'] = traceback.format_exc()
        finally:
            with _lock:
                _jobs[job_id]['done'] = True
                _jobs[job_id]['status'] = 'done'

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/poll/<job_id>')
def poll(job_id):
    since = int(request.args.get('since', 0))
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'status': job['status'],
        'events': job['events'][since:],
        'total': len(job['events']),
        'current': job['current'],
        'done': job['done'],
        'error': job['error'],
        'gp_keys': list(job['gp_maps'].keys()),
    })


@app.route('/api/gp_map/<job_id>')
def gp_map(job_id):
    method = request.args.get('method', 'A')
    seed   = int(request.args.get('seed', 0))
    it     = int(request.args.get('iter', 1))
    key    = f"{method}_{seed}_{it}"
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    data = job['gp_maps'].get(key)
    if not data:
        candidates = [(k, int(k.split('_')[2]))
                      for k in job['gp_maps'] if k.startswith(f"{method}_{seed}_")]
        if candidates:
            closest = min(candidates, key=lambda t: abs(t[1] - it))[0]
            data = job['gp_maps'][closest]
    if not data:
        return jsonify({'error': 'Not available yet'}), 404
    return jsonify(data)


@app.route('/api/landscape_real')
def landscape_real():
    from data_model.imod_oracle import load_pool
    donor_qw  = request.args.get('donor_qw',  'QW1')
    target_qw = request.args.get('target_qw', 'QW2')
    X_pool, y_donor_pool, y_target_pool, X_donor_np, y_donor_train = load_pool(
        None, donor_qw, target_qw
    )
    return jsonify({
        'donor': {
            'r_mai':  X_donor_np[:, 2].tolist(),
            'r_baac': X_donor_np[:, 3].tolist(),
            'score':  y_donor_train.tolist(),
        },
        'target': {
            'r_mai':  X_pool[:, 2].tolist(),
            'r_baac': X_pool[:, 3].tolist(),
            'score':  y_target_pool.tolist(),
            'donor_score': y_donor_pool.tolist(),
        },
        'donor_qw':  donor_qw,
        'target_qw': target_qw,
    })


@app.route('/api/qw_cols')
def qw_cols():
    from data_model.imod_oracle import QW_COLS
    return jsonify({'qw_cols': QW_COLS})


@app.route('/api/run_real', methods=['POST'])
def run_real():
    from bo_runner_real import run_experiment_real
    config = request.json
    job_id = str(uuid.uuid4())[:8]

    job = {
        'status': 'running',
        'config': config,
        'events': [],
        'gp_maps': {},
        'done': False,
        'error': None,
        'current': {'method': None, 'seed': 0, 'iter': 0}
    }
    with _lock:
        _jobs[job_id] = job

    def on_event(event):
        with _lock:
            j = _jobs[job_id]
            gp_map = event.pop('gp_map', None)
            j['events'].append(dict(event))
            j['current'] = {
                'method': event['method'],
                'seed': event['seed'],
                'iter': event['iter']
            }
            if gp_map:
                key = f"{event['method']}_{event['seed']}_{event['iter']}"
                j['gp_maps'][key] = gp_map

    def worker():
        try:
            run_experiment_real(config, on_event)
        except Exception:
            import traceback
            with _lock:
                _jobs[job_id]['error'] = traceback.format_exc()
        finally:
            with _lock:
                _jobs[job_id]['done'] = True
                _jobs[job_id]['status'] = 'done'

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id})


if __name__ == '__main__':
    app.run(threaded=True, port=5050, use_reloader=False)
