--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--
require 'image'
local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.split = split
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      _G.preprocessRandCrop = dataset:preprocessRandCrop()
      _G.preprocessHFlip = dataset:preprocessHFlip()
      _G.preprocessVFlip = dataset:preprocessVFlip()
      _G.preprocessColorJitter = dataset:preprocessColorJitter()

      _G.preprocessRotation = dataset:preprocessRotation()      
      _G.preprocessScale = dataset:preprocessScale()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.split = split
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.crop_num = opt.crop_num or 0
   self.hflip_num = opt.hflip_num or 0
   self.vflip_num = opt.vflip_num or 0
   self.color_jitter_num = opt.color_jitter_num or 0
   self.rotation_num = opt.rotation_num or 0
   self.scale_num = opt.scale_num or 0
   self.aug_num = self.crop_num + self.hflip_num + self.vflip_num + self.color_jitter_num + self.rotation_num + self.scale_num
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run(get_type)
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)
               
   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         
         threads:addjob(
            function(indices, nCrops, crop_num, hflip_num, vflip_num, color_jitter_num, rotation_num, scale_num)
               local sz = indices:size(1)
               local target = torch.IntTensor(sz)
               
               local buffer = {}
               local batch_count = 0
               --Generate crop set
               for _ = 1, crop_num do
                 local batch, imageSize               
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessRandCrop(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end
               if _G.split == 'val' then 
                  collectgarbage()
                  return buffer 
               end
               --Generate horizontal flip set
               for _ = 1, hflip_num do
                 local batch, imageSize 
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessHFlip(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end
               
               --Generate vertical flip set
               for _ = 1, vflip_num do
                 local batch, imageSize 
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessVFlip(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end
               
               --Generate color jittering set
               for _ = 1, color_jitter_num do
                 local batch, imageSize 
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessColorJitter(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end               
               
               --Generate rotation set
               for _ = 1, rotation_num do
                 local batch, imageSize 
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessRotation(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end   
               
               --Generate scale set
               for _ = 1, scale_num do
                 local batch, imageSize 
                 for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input = _G.preprocessScale(sample.input)
                    if not batch then
                       imageSize = input:size():totable()
                       if nCrops > 1 then table.remove(imageSize, 1) end
                       batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                 end
                 batch_count = batch_count + 1
                 buffer[batch_count] = {
                    input = batch:view(sz * nCrops, table.unpack(imageSize)),
                    target = target,
                 }
               end
               collectgarbage()
               --if batch_count ~= self.aug_num then error("Augmentation number doesn't match!") end
               return buffer
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            self.crop_num,
            self.hflip_num,
            self.vflip_num,
            self.color_jitter_num,
            self.rotation_num,
            self.scale_num
         )
         idx = idx + batchSize
      end
   end

   local n_train = 0
   local n_val = 0
   local buffer_idx = 0
   local function loop()
      if n_train % self.aug_num == 0 or self.split == 'val' then
        enqueue()
        if not threads:hasjob() then
           return nil
        end
        threads:dojob()
        if threads:haserror() then
           threads:synchronize()
        end
      end
      buffer_idx = -1
      if self.split == 'val' then
         n_val = n_val + 1
         buffer_idx = 1
      else
         n_train = n_train + 1
         buffer_idx = n_train % self.aug_num == 0 and self.aug_num or n_train % self.aug_num
      end
      assert(buffer_idx ~= -1)
      return n_train + n_val, sample[buffer_idx]
   end

   return loop
end

return M.DataLoader
