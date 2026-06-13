import io
import base64
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, send_file, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_uploaded_file(file_storage, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        return pd.read_csv(file_storage)
    elif ext == 'tsv' or ext == 'txt':
        return pd.read_csv(file_storage, sep='\t')
    else:
        return pd.read_csv(file_storage)


def handle_missing_values(numeric_df, strategy='drop_rows'):
    total_cells = numeric_df.size
    missing_cells = numeric_df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

    missing_per_column = numeric_df.isnull().sum().to_dict()

    if strategy == 'drop_rows':
        cleaned_df = numeric_df.dropna(axis=0, how='any')
    elif strategy == 'drop_cols':
        cleaned_df = numeric_df.dropna(axis=1, how='any')
    elif strategy == 'mean':
        cleaned_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
    elif strategy == 'median':
        cleaned_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    elif strategy == 'zero':
        cleaned_df = numeric_df.fillna(0)
    else:
        cleaned_df = numeric_df.dropna(axis=0, how='any')

    rows_before = numeric_df.shape[0]
    cols_before = numeric_df.shape[1]
    rows_after = cleaned_df.shape[0]
    cols_after = cleaned_df.shape[1]

    stats = {
        'total_missing_cells': int(missing_cells),
        'missing_ratio': round(missing_ratio, 4),
        'missing_per_column': {k: int(v) for k, v in missing_per_column.items()},
        'strategy': strategy,
        'rows_before': int(rows_before),
        'rows_after': int(rows_after),
        'rows_removed': int(rows_before - rows_after),
        'cols_before': int(cols_before),
        'cols_after': int(cols_after),
        'cols_removed': int(cols_before - cols_after),
    }

    return cleaned_df, stats


def compute_correlation(df, missing_strategy='drop_rows'):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError('数据集中至少需要两列数值型数据才能计算相关系数')

    cleaned_df, missing_stats = handle_missing_values(numeric_df, strategy=missing_strategy)

    if cleaned_df.shape[1] < 2:
        raise ValueError(f'缺失值处理后仅剩 {cleaned_df.shape[1]} 列数值型数据，请尝试其他缺失值处理策略（至少需要2列）')

    if cleaned_df.shape[0] < 2:
        raise ValueError(f'缺失值处理后仅剩 {cleaned_df.shape[0]} 行数据，请尝试其他缺失值处理策略（至少需要2行）')

    corr_matrix = cleaned_df.corr()
    return corr_matrix, missing_stats


def generate_heatmap(df, annot=True, fmt='.2f', cmap='coolwarm', title='Correlation Heatmap', missing_strategy='drop_rows'):
    corr_matrix, missing_stats = compute_correlation(df, missing_strategy=missing_strategy)

    fig, ax = plt.subplots(figsize=(max(10, corr_matrix.shape[1] * 0.8), max(8, corr_matrix.shape[1] * 0.7)))
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        ax=ax
    )
    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)

    return img_buffer, corr_matrix, missing_stats


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传的文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传 CSV、TSV 或 TXT 文件'}), 400

    try:
        df = read_uploaded_file(file, file.filename)

        annot = request.form.get('annot', 'true').lower() == 'true'
        cmap = request.form.get('cmap', 'coolwarm')
        title = request.form.get('title', 'Correlation Heatmap')
        missing_strategy = request.form.get('missing_strategy', 'drop_rows')

        img_buffer, corr_matrix, missing_stats = generate_heatmap(
            df, annot=annot, cmap=cmap, title=title, missing_strategy=missing_strategy
        )

        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'shape': list(corr_matrix.shape),
            'columns': list(corr_matrix.columns),
            'correlation_matrix': corr_matrix.to_dict(),
            'missing_stats': missing_stats
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500


@app.route('/upload/download', methods=['POST'])
def upload_download():
    if 'file' not in request.files:
        return '未找到上传的文件', 400

    file = request.files['file']
    if file.filename == '':
        return '未选择文件', 400

    if not allowed_file(file.filename):
        return '不支持的文件格式', 400

    try:
        df = read_uploaded_file(file, file.filename)

        annot = request.form.get('annot', 'true').lower() == 'true'
        cmap = request.form.get('cmap', 'coolwarm')
        title = request.form.get('title', 'Correlation Heatmap')
        missing_strategy = request.form.get('missing_strategy', 'drop_rows')

        img_buffer, _, _ = generate_heatmap(
            df, annot=annot, cmap=cmap, title=title, missing_strategy=missing_strategy
        )

        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='correlation_heatmap.png'
        )

    except Exception as e:
        return str(e), 500


@app.route('/api/correlation', methods=['POST'])
def api_correlation():
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传的文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传 CSV、TSV 或 TXT 文件'}), 400

    try:
        df = read_uploaded_file(file, file.filename)

        missing_strategy = request.form.get('missing_strategy', 'drop_rows')
        method = request.form.get('method', 'pearson')

        corr_matrix, missing_stats = compute_correlation(df, missing_strategy=missing_strategy)

        columns = list(corr_matrix.columns)
        matrix_values = []
        for i, row in enumerate(columns):
            for j, col in enumerate(columns):
                matrix_values.append({
                    'row': row,
                    'col': col,
                    'value': round(float(corr_matrix.iloc[i, j]), 4)
                })

        return jsonify({
            'success': True,
            'method': method,
            'columns': columns,
            'shape': list(corr_matrix.shape),
            'matrix': matrix_values,
            'matrix_2d': [[round(float(v), 4) for v in row] for row in corr_matrix.values],
            'correlation_matrix': corr_matrix.to_dict(),
            'missing_stats': missing_stats
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
