library(DNAcopy)

CBS_data <- function(){
  args <- commandArgs(T)
  name = paste(args[1], "txt", sep=".")
  print(name)
  data = read.table(name)

  # 检查数据是否包含NA值
  if (any(is.na(data))) {
    stop("Error: Input data contains NA values. Please clean the data.")
  }

  # 打印数据结构，确保数据格式正确
  print(str(data))

  # 数据平滑处理
  head = matrix(0, nrow(data), 3)
  head[,1] = 1
  head[,2] = 1:nrow(data)
  head[,3] = 1:nrow(data)

  chrom <- data$V1
  maploc <- 1:nrow(data)
  seg.file_g = matrix(0, 1, 6)
  seg.file_g_one = matrix(0, 1, 6)
  seg.file = matrix(0, nrow(data), 1)

  stac_amp = matrix(0, 1, nrow(data))
  stac_amp[1,] = 1:nrow(data)
  stac_amp_one = matrix(0, 1, nrow(data))

  stac_del = matrix(0, 1, nrow(data))
  stac_del[1,] = 1:nrow(data)
  stac_del_one = matrix(0, 1, nrow(data))

  # 进行数据分段
  for (j in 1:ncol(data)) {
    smoothed.data <- smooth.CNA(CNA(data[, j], chrom, maploc))
    seg <- segment(smoothed.data)
    for (k in 1:length(seg$output$loc.start)) {
      seg.file_g_one[1,1] = j
      seg.file_g_one[1,2] = 1
      seg.file_g_one[1,3] = seg$output$loc.start[k]
      seg.file_g_one[1,4] = seg$output$loc.end[k]
      seg.file_g_one[1,5] = seg$output$num.mark[k]
      seg.file_g_one[1,6] = seg$output$seg.mean[k]
      seg.file_g = rbind(seg.file_g, seg.file_g_one)
      seg.file_g_one = matrix(0, 1, 6)
    }
  }

  seg.file_g = seg.file_g[-1,]  # 去掉初始化时的空行

  # 获取当前工作目录
  out.file = getwd()  # 获取当前工作目录
  out.file = paste(out.file, 'seg', sep="/")  # 生成seg文件路径
  print(paste("Writing output to: ", out.file))

  # 确保目标文件夹存在，如果不存在则创建
  if (!dir.exists(dirname(out.file))) {
    dir.create(dirname(out.file), recursive = TRUE)
    print(paste("Created directory:", dirname(out.file)))
  }

  # 确保seg.file_g包含数据
  if (nrow(seg.file_g) == 0) {
    stop("Error: seg.file_g is empty. No data to write.")
  }

  # 打印seg.file_g的前几行，确认数据是否被正确填充
  print(head(seg.file_g))

  # 检查当前工作目录是否有写权限
  if (!file.access(getwd(), 2) == 0) {
    stop("Error: No write permission to the directory.")
  }

  # 尝试写入seg文件
  tryCatch({
    write.table(seg.file_g, file=out.file, row.names=F, col.names=F, quote=F, sep="\t")
    print(paste("Seg file successfully written to:", out.file))
  }, error = function(e) {
    print(paste("Error writing the seg file:", e$message))
  })
}

CBS_data()
