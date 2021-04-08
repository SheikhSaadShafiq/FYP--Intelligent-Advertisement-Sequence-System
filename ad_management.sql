-- phpMyAdmin SQL Dump
-- version 5.0.4
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 29, 2021 at 01:23 PM
-- Server version: 10.4.17-MariaDB
-- PHP Version: 8.0.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ad_management`
--

-- --------------------------------------------------------

--
-- Table structure for table `ad`
--

CREATE TABLE `ad` (
  `AD_Code` int(11) NOT NULL,
  `AdName` varchar(15) DEFAULT NULL,
  `BrandName` varchar(15) DEFAULT NULL,
  `Gender` varchar(15) DEFAULT NULL,
  `Time_duration` time DEFAULT NULL,
  `Priority` int(11) DEFAULT NULL,
  `Priority_time` time DEFAULT NULL,
  `AD_Status` varchar(15) DEFAULT NULL,
  `Manager_ID` int(11) DEFAULT NULL,
  `File_path` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `ad`
--

INSERT INTO `ad` (`AD_Code`, `AdName`, `BrandName`, `Gender`, `Time_duration`, `Priority`, `Priority_time`, `AD_Status`, `Manager_ID`, `File_path`) VALUES
(1, 'CBR250', 'Honda', '0', '00:01:00', 1, '00:12:00', 'active', 1, 'C:/Users/Sanan\'s PC/Desktop/pyfront/img/aorus-logo'),
(2, 'hair dye', 'loreal', '1', '00:00:30', 1, '00:15:00', 'inactive', 2, 'C:/Users/Sanan\'s PC/Desktop/pyfront/img/append.JPG'),
(3, 'loafers', 'hushpuppy', '1', '00:00:15', 0, '00:18:00', 'active', 3, 'C:/Users/Sanan\'s PC/Desktop/pyfront/img/log.JPG'),
(4, 'washing powder', 'ariel', '1', '00:01:15', 0, '00:16:00', 'active', 2, 'C:/Users/Sanan\'s PC/Videos/Mulan.2020.1080p.WEBRip'),
(5, 'denim jeans', 'levis', '0', '00:01:10', 0, '00:21:00', 'active', 1, 'C:/Users/Sanan\'s PC/Pictures/Saved Pictures/triang'),
(6, 'concealer', 'mac', '0', '00:00:30', 1, '00:12:00', 'active', 3, NULL),
(7, 'lux', 'uniliver', '0', '00:00:15', 1, '00:15:00', 'active', 3, 'C:/Users/Sanan\'s PC/Desktop/pyfront/img/delete.JPG'),
(8, 'lawn', 'sapphire', '0', '00:00:15', 1, '00:15:00', 'active', 4, 'C:/Users/Sanan\'s PC/Desktop/pyfront/back.JPG');

-- --------------------------------------------------------

--
-- Table structure for table `manager`
--

CREATE TABLE `manager` (
  `ID` int(11) NOT NULL,
  `username` varchar(20) DEFAULT NULL,
  `password` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `manager`
--

INSERT INTO `manager` (`ID`, `username`, `password`) VALUES
(1, 'sanan@ucp.edu.pk', 'sa123123'),
(2, 'gauher@ucp.edu.pk', 'ga321321'),
(3, 'saad@ucp.edu.pk', 'saad_123'),
(4, 'humza@gmail.com', 'asd12345'),
(5, 'ali@gmail.com', 'ali12345'),
(6, 'ahmed@ucp.edu.pk', 'ahmed123');

-- --------------------------------------------------------

--
-- Table structure for table `played_list`
--

CREATE TABLE `played_list` (
  `AD_Code` int(11) NOT NULL,
  `played_date` date NOT NULL,
  `No_of_Times` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `ad`
--
ALTER TABLE `ad`
  ADD PRIMARY KEY (`AD_Code`),
  ADD KEY `Manager_ID` (`Manager_ID`);

--
-- Indexes for table `manager`
--
ALTER TABLE `manager`
  ADD PRIMARY KEY (`ID`);

--
-- Indexes for table `played_list`
--
ALTER TABLE `played_list`
  ADD PRIMARY KEY (`AD_Code`,`played_date`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `ad`
--
ALTER TABLE `ad`
  MODIFY `AD_Code` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `manager`
--
ALTER TABLE `manager`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `ad`
--
ALTER TABLE `ad`
  ADD CONSTRAINT `ad_ibfk_1` FOREIGN KEY (`Manager_ID`) REFERENCES `manager` (`ID`);

--
-- Constraints for table `played_list`
--
ALTER TABLE `played_list`
  ADD CONSTRAINT `played_list_fk` FOREIGN KEY (`AD_Code`) REFERENCES `ad` (`AD_Code`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
