/*
  Warnings:

  - Made the column `image` on table `Blogs` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "Blogs" ALTER COLUMN "image" SET NOT NULL,
ALTER COLUMN "image" SET DEFAULT 'http://choseandclick.com/image/cache/catalog/basel-demo/blog-1140x700.png';
