# process.py

## 1. Usage
+ **class Process**

```python
from process import Process
from process import HashProcess

process = Process()

process.feature_param = Process.Feature()
process.weight_param = Process.Weight()
process.distribution_param = Process.Distribution()

    # Above code is same to this code
    # process.feature_param = Process.Feature(Process.Feature.fill_all)
    # process.weight_param = Process.Weight(Process.Weight.no_weight_05)
    # process.distribution_param = Process.Distribution(
    #     drop_mail_order=False,
    #     fnb_related=False
    # )

commercial_data_path = '{your commercial data path! ex: data/updated_diningcode_youngdeungpo_20241.csv}'
local_data_path = '{your local data path! ex: data/final_merged_filtered_youngdeungpo_data_20241.csv}'
result_data_path = '{your result data path! ex: data/inter_diningcode_youngdeungpo_dropped.csv}'
process.process_data(commercial_data_path, local_data_path, result_data_path)
```

결과: **result_data_path** 로 지정한 directory로 process된 파일 생성
+ **class HashProcess**

```python
from process import Process
from process import HashProcess

hash_process = HashProcess()

hash_process.feature_param = HashProcess.Feature()
hash_process.weight_param = HashProcess.Weight()

   
    # Above code is same to this code
    # hash_process.feature_param = HashProcess.Feature(HashProcess.Feature.no_feature)
    # hash_process.weight_param = HashProcess.Weight(HashProcess.Weight.no_weight_05)
    # hash_process.distribution_param = HashProcess.Distribution(
    #     self.todo!() = False
    # )


commercial_data_path = '{your commercial data path! ex: data/updated_diningcode_youngdeungpo_20241.csv}'
local_data_path = '{your local data path! ex: data/updated_diningcode_youngdeungpo_20241.csv}'
result_data_path = '{your result data path! ex: data/inter_hashes_diningcode_youngdeungpo_dropped.csv}'
hash_process.process_data(commercial_data_path, local_data_path, result_data_path)
```

결과: **result_data_path** 로 지정한 directory로 process된 파일 생성

## 2. Code

+ class **Process**
	+ class **Feature**
		
        지역 데이터(local_data) 의 특정 column을 업데이트 합니다. feature는 함수의 형태로 self.feature_param에 저장되므로, 다음과 같이 사용할 수 있습니다.
        
        ```python
        data_chunk = self.feature_param.modify_feature(chunk)
        ```
        
		**fill_all**: 업태구분명의 빈칸을 '개방서비스명' 으로 대체시킵니다.
    	
        **drop_unrelated**: 중요도가 높은 특정 개방서비스명에 해당하는 경우만 업태구분명을 유지시킵니다. 그 외의 업태구분명(중요도가 낮은 업태구분명)은, 개방서비스명으로 대체합니다.
        
	+ class **Weight**
		
        거리에 따른 데이터의 가중치를 설정합니다. weight는 함수의 형태로 self.weight_param에 저장되므로, 다음과 같이 사용할 수 있습니다.
        
        ```python
        data['weight'] = np.where(mask, self.weight_param.calculate_weight(distances=distances), 0)
        ```
        
		**circular_weight**: 1km 이내의 원형 가중치를 부여합니다.
    	
        **gaussian_weight**: 1km 이내의 가우시안 분포에 따른 가중치를 부여합니다.
    	
        **no_weight_05**: 가중치를 부여하지 않으며, 0.5km이내만 탐색합니다.
    	
        **double_weight_05**: 0.5km 이내의 건물에 2의 가중치를 부여하며, 1km 이내는 1의 가중치를 부여합니다.
        
	+ class **Distribution**
		
        위의 두 class에 해당되지 않는 항목을 **attribute**로 추가하고, 수정할 수 있습니다.distribution class 의 attribute는 동시에 여러 개 적용시킬 수 있습니다.
        
        class Distribution내에서 참인 attribute만 적용됩니다. 아래의 함수로, 참인 attribute의 list를 얻을 수 있습니다.
        ```python
        self.feature_param.get_true_attributes()
        ```
        특정 attribute의 bool 값을 바꾸고 싶다면, 아래의 함수에 attribute의 이름, bool 값을 인자로 넣어 attribute를 수정할 수 있습니다.
        ```python
        self.feature_param.set_value(self, attribute_name, value)
        ```
        
		self.**drop_mail.order**: 업태구분명이 '통신판매업'인 경우, drop합니다.
        
		self.**fnb_related**: 중요도가 높은 특정 개방서비스명에 해당하는 경우에, 가중치를 2배로 합니다.
    
    그 외의 feature engineering이 필요하다면, 따로 추가할 수 있습니다.

____

+ class **HashProcess**
	+ class **Feature**
		
        지역 데이터(local_data) 의 특정 column을 업데이트 합니다. 지역 데이터는 상권 데이터(commercial_data)와 동일한 것을 사용합니다.
        
		**no_feature**: 지역데이터의 hashes를 각각 hash로 구분하여, 데이터 내에 복제합니다.
        
	+ class **Weight**
		
        거리에 따른 데이터의 가중치를 설정합니다.
        
		**circular_weight**: 1km 이내의 원형 가중치를 부여합니다.
    	
        **gaussian_weight**: 1km 이내의 가우시안 분포에 따른 가중치를 부여합니다.
    	
        **no_weight_05**: 가중치를 부여하지 않으며, 0.5km이내만 탐색합니다.
    	
        **double_weight_05**: 0.5km 이내의 건물에 2의 가중치를 부여하며, 1km 이내는 1의 가중치를 부여합니다.
        
	+ class **Distribution**
		
        HashProcess에서 사용하지 않습니다. (사용이 필요하다면, 추가할 수 있습니다.)
    
    그 외의 feature engineering이 필요하다면, 따로 추가할 수 있습니다.
    
____

+ Utility function
	+ **__vectorized_haversine**(self, lat1, lon1, lat2_array, lon2_array)
	
		위도, 경도, 위도 벡터, 경도 벡터를 입력받아, 거리에 대한 벡터를 계산합니다.
        
	+ **get_weighted_distribution**(self, row_tuple, data, radius=1.0)
	
		class **Weight**, class **Distribution**을 적용시켜, 가중치를 계산합니다.
	+ **__read_data**(self, commercial_data_path, local_data_path)
	
		commercial data, local data를 읽어 data frame으로 저장하고, class **Feature**를 적용시킵니다.
	+ **process_data**(self, commercial_data_path, local_data_path, result_data_path)
	
		process = Process()로 불러온 이후, class Weight, Distribution, Feature를 설정한 이후 실행시키면, 각 class를 적용시켜 모든 과정을 자동으로 진행합니다. 정확한 사용은, *process.md* 상단의 **Usage** 를 참고하세요.
    
    + **get**_{**weight**/**distribution**/**feature**}_**param**
    
    	각 class에서 적용시킬 수 있는 모든 attribute/ function 을 리스트의 형태로 가져옵니다.
        
```python
from process import Process
from process import HashProcess

# feature_param_list and weight_param_list is list of function
# feature_param_list = [fill_all(), drop_unrelated(), ...]
feature_param_list = process.get_feature_param()

# weight_param_list = [no_weight_05(), gaussian()...]
weight_param_list = process.get_weight_param()

# distribution_param_list is list of attributes
# distribution_param_list = ['drop_mail_order', 'fnb_related', ...]
distribution_param_list = process.get_distribution_param()
```
