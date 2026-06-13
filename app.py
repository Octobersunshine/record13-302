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


def generate_heatmap(df, annot=True, fmt='.2f', cmap='coolwarm', title='Correlation Heatmap'):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError('数据集中至少需要两列数值型数据才能计算相关系数')

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(max(10, numeric_df.shape[1] * 0.8), max(8, numeric_df.shape[1] * 0.7)))
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

    return img_buffer, corr_matrix


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

        img_buffer, corr_matrix = generate_heatmap(df, annot=annot, cmap=cmap, title=title)

        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'shape': list(corr_matrix.shape),
            'columns': list(corr_matrix.columns),
            'correlation_matrix': corr_matrix.to_dict()
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

        img_buffer, _ = generate_heatmap(df, annot=annot, cmap=cmap, title=title)

        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='correlation_heatmap.png'
        )

    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
