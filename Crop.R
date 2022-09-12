
#Load orthophoto
ortho<- brick('///////////')
traindir<-"/////////////"
maskdir<-"//////////////"

shp<- shapefile('//////////////')

for (i in 1:length(shp)){ 
  inst <- subset(shp,shp$id==i)
  centre <- centroid(inst)
  footprint <- raster(resolution=res(ortho),extent(centre[1,1]-6,centre[1,1]+6,centre[1,2]-6,centre[1,2]+6))
  train_img <- crop(ortho,footprint)
  train_mask <- crop(shp, extent(footprint))
  train_mask<- rasterize(train_mask,footprint,field=as.integer(train_mask$id)+254,background=0)
  #export training data set
  writeGDAL(as(train_img, Class ="SpatialGridDataFrame"), fname = paste(traindir,"train",i,".png",sep=""), drivername = "PNG")#,type = "Byte", mvFlag = 255)
  writeGDAL(as(train_mask, Class ="SpatialGridDataFrame"), fname = paste(maskdir,"mask",i,".png",sep=""), drivername = "PNG")#,type = "Byte", mvFlag = 255)
  gc()
}

