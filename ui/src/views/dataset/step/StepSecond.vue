<template>
  <div class="set-rules">
    <el-row>
      <el-col :span="10" class="p-24">
        <div class="scrollbar-height-left">
          <el-scrollbar>
            <div class="set-rules__right">
              <div>
                <h4 class="title-decoration-1 mb-16">raptop模型</h4>
                <el-select
                  v-model="raptor_model_id"
                  placeholder="若使用raptor, 请设置AI模型"
                  class="w-full"
                  popper-class="select-model"
                  :clearable="true"
                >
                  <el-option-group
                    v-for="(value, label) in modelOptions"
                    :key="value"
                    :label="relatedObject(providerOptions, label, 'provider')?.name"
                  >
                    <el-option
                      v-for="item in value.filter((v: any) => v.status === 'SUCCESS')"
                      :key="item.id"
                      :label="item.name"
                      :value="item.id"
                      class="flex-between"
                    >
                      <div class="flex">
                        <span
                          v-html="relatedObject(providerOptions, label, 'provider')?.icon"
                          class="model-icon mr-8"
                        ></span>
                        <span>{{ item.name }}</span>
                      </div>
                      <el-icon class="check-icon" v-if="item.id === raptor_model_id"
                        ><Check
                      /></el-icon>
                    </el-option>
                    <!-- 不可用 -->
                    <el-option
                      v-for="item in value.filter((v: any) => v.status !== 'SUCCESS')"
                      :key="item.id"
                      :label="item.name"
                      :value="item.id"
                      class="flex-between"
                      disabled
                    >
                      <div class="flex">
                        <span
                          v-html="relatedObject(providerOptions, label, 'provider')?.icon"
                          class="model-icon mr-8"
                        ></span>
                        <span>{{ item.name }}</span>
                        <span class="danger">（不可用）</span>
                      </div>
                      <el-icon class="check-icon" v-if="item.id === raptor_model_id"
                        ><Check
                      /></el-icon>
                    </el-option>
                  </el-option-group>
                </el-select>
              </div>
              <div class="empty-line"></div>
              <h4 class="title-decoration-1 mb-16">设置分段规则</h4>
              <el-scrollbar>
                <div class="left-height" @click.stop>
                  <el-radio-group v-model="radio" class="set-rules__radio">
                    <el-card shadow="never" class="mb-16" :class="radio === '1' ? 'active' : ''">
                      <el-radio value="1" size="large">
                        <p class="mb-4">智能分段（推荐）</p>
                        <el-text type="info">不了解如何设置分段规则推荐使用智能分段</el-text>
                      </el-radio>
                    </el-card>
                    <el-card shadow="never" class="mb-16" :class="radio === '2' ? 'active' : ''">
                      <el-radio value="2" size="large">
                        <p class="mb-4">高级分段</p>
                        <el-text type="info"
                          >用户可根据文档规范自行设置分段标识符、分段长度以及清洗规则
                        </el-text>
                      </el-radio>

                      <el-card
                        v-if="radio === '2'"
                        shadow="never"
                        class="card-never mt-16"
                        style="margin-left: 30px"
                      >
                        <div class="set-rules__form">
                          <div class="form-item mb-16">
                            <div class="title flex align-center mb-8">
                              <span style="margin-right: 4px">分段标识</span>
                              <el-tooltip
                                effect="dark"
                                content="通用：按照所选符号先后顺序做递归分割，分割结果超出分段长度将截取至分段长度。
                                PDF：可以选择OCR模式进行识别（默认为pptx库）。
                                EXCEL：可以选择markdown格式进行分段（默认为html）。"
                                placement="right"
                              >
                                <AppIcon iconName="app-warning" class="app-warning-icon"></AppIcon>
                              </el-tooltip>
                            </div>
                            <div @click.stop>
                              <el-select
                                v-model="form.patterns"
                                multiple
                                allow-create
                                default-first-option
                                filterable
                                placeholder="请选择"
                              >
                                <el-option
                                  v-for="(item, index) in splitPatternList"
                                  :key="index"
                                  :label="item.key"
                                  :value="item.value"
                                >
                                </el-option>
                              </el-select>
                            </div>
                          </div>
                          <div class="form-item mb-16">
                            <div class="title mb-8">分段长度（当文档为EXCEL时, 限制的是行数）</div>
                            <el-slider
                              v-model="form.limit"
                              show-input
                              :show-input-controls="false"
                              :min="0"
                              :max="4096"
                            />
                          </div>
                          <div class="form-item mb-16">
                            <div class="title flex align-center mb-8">
                              <span style="margin-right: 4px">自动清洗</span>
                              <el-tooltip
                                  effect="dark"
                                  content="去掉重复多余符号空格、空行、制表符"
                                  placement="right"
                                >
                                <AppIcon iconName="app-warning" class="app-warning-icon"></AppIcon>
                              </el-tooltip>
                            </div>
                            <el-switch size="small" v-model="form.with_filter" />
                          </div>
                        </div>
                      </el-card>
                    </el-card>
                  </el-radio-group>
                </div>
              </el-scrollbar>
              <div>
                <el-checkbox v-model="checkedConnect" @change="changeHandle">
                  导入时添加分段标题为关联问题（适用于标题为问题的问答对）
                </el-checkbox>
              </div>
              <div class="text-right mt-8">
                <el-button @click="splitDocument">生成预览</el-button>
              </div>
            </div>
          </el-scrollbar>
        </div>
      </el-col>

      <el-col :span="14" class="p-24 border-l">
        <div v-loading="loading">
          <h4 class="title-decoration-1 mb-8">分段预览</h4>

          <ParagraphPreview v-model:data="paragraphList" :isConnect="checkedConnect" />
        </div>
      </el-col>
    </el-row>
  </div>
</template>
<script setup lang="ts">
import { ref, computed, onMounted, reactive, watch } from 'vue'
import ParagraphPreview from '@/views/dataset/component/ParagraphPreview.vue'
import documentApi from '@/api/document'
import useStore from '@/stores'
import type { KeyValue } from '@/api/type/common'
import { relatedObject } from '@/utils/utils'
import { groupBy } from 'lodash'
const { dataset } = useStore()
const documentsFiles = computed(() => dataset.documentsFiles)
const splitPatternList = ref<Array<KeyValue<string, string>>>([])

const radio = ref('1')
const loading = ref(false)
const paragraphList = ref<any[]>([])
const patternLoading = ref<boolean>(false)
const checkedConnect = ref<boolean>(false)
const raptor_model_id = ref('')

const firstChecked = ref(true)

const form = reactive<{
  patterns: Array<string>
  limit: number
  with_filter: boolean
  [propName: string]: any
}>({
  patterns: [],
  limit: 500,
  with_filter: true
})

const { model } = useStore()
const modelOptions = ref<any>(null)
const providerOptions = ref([])


function getModel() {
  loading.value = true
  model
    .asyncGetModel()
    .then((res: any) => {
      modelOptions.value = groupBy(res?.data, 'provider')
      loading.value = false
    })
    .catch(() => {
      loading.value = false
    })
}

function getProvider() {
  loading.value = true
  model
    .asyncGetProvider()
    .then((res: any) => {
      providerOptions.value = res?.data
      loading.value = false
    })
    .catch(() => {
      loading.value = false
    })
}

function changeHandle(val: boolean) {
  if (val && firstChecked.value) {
    const list = paragraphList.value
    list.map((item: any) => {
      item.content.map((v: any) => {
        v['problem_list'] = v.title.trim()
          ? [
              {
                content: v.title.trim()
              }
            ]
          : []
      })
    })
    paragraphList.value = list
    firstChecked.value = false
  }
}

function splitDocument() {
  loading.value = true
  let fd = new FormData()
  documentsFiles.value.forEach((item) => {
    if (item?.raw) {
      fd.append('file', item?.raw)
    }
  })
  if (radio.value === '2') {
    Object.keys(form).forEach((key) => {
      if (key == 'patterns') {
        form.patterns.forEach((item) => fd.append('patterns', item))
      } else {
        fd.append(key, form[key])
      }
    })
  }
  documentApi
    .postSplitDocument(fd)
    .then((res: any) => {
      const list = res.data
      if (checkedConnect.value) {
        list.map((item: any) => {
          item.content.map((v: any) => {
            v['problem_list'] = v.title.trim()
              ? [
                  {
                    content: v.title.trim()
                  }
                ]
              : []
          })
        })
      }
      list.forEach((item: any) => { 
        item.raptor_model = raptor_model_id;
      })
      paragraphList.value = list
      loading.value = false
    })
    .catch(() => {
      loading.value = false
    })
}

const initSplitPatternList = () => {
  documentApi.listSplitPattern(patternLoading).then((ok) => {
    splitPatternList.value = ok.data
  })
}

watch(radio, () => {
  if (radio.value === '2') {
    initSplitPatternList()
  }
})

onMounted(() => {
  getProvider()
  getModel()
  splitDocument()
})

defineExpose({
  paragraphList,
  checkedConnect
})
</script>
<style scoped lang="scss">
.set-rules {
  width: 100%;

  .left-height {
    max-height: calc(var(--create-dataset-height) - 110px);
    overflow-x: hidden;
  }

  &__radio {
    width: 100%;
    display: block;

    .el-radio {
      white-space: break-spaces;
      width: 100%;
      height: 100%;
      line-height: 22px;
      color: var(--app-text-color);
    }

    :deep(.el-radio__label) {
      padding-left: 30px;
      width: 100%;
    }
    :deep(.el-radio__input) {
      position: absolute;
      top: 16px;
    }
    .active {
      border: 1px solid var(--el-color-primary);
    }
  }

  &__form {
    .title {
      font-size: 14px;
      font-weight: 400;
    }
  }
}
.model-icon {
  width: 20px;
}
.check-icon {
  position: absolute;
  right: 10px;
}
.empty-line {
  height: 16px; 
}
.scrollbar-height-left {
    height: calc(var(--app-main-height) - 127px);
  }
</style>
