Dataset **VisDrone2019-VID** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](Set 'HIDE_DATASET=False' to generate download link)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='VisDrone2019-VID', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be downloaded here:

- [trainset](https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw/view?usp=sharing)
- [valset](https://drive.google.com/file/d/1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B/view?usp=sharing)
- [testset-dev](https://drive.google.com/open?id=1-BEq--FcjshTF1UwUabby_LHhYj41os5)
- [testset-challenge](https://drive.google.com/file/d/1Qwyp_cEpGyXGqJ8IbusEzuNHgbM403NP/view?usp=sharing)
