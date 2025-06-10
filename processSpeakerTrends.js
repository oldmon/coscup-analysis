/**
 * 處理講者趨勢數據，分析各年度講者活動情況
 * Process speaker trend data and analyze speaker activities across years
 * 
 * @param {Object} yearlyData - 年度資料物件，包含各年度的議程和講者資訊
 *                             Yearly data object containing sessions and speaker information
 * @param {number} topNSpeaker - 要返回的前N名最活躍講者數量
 *                               Number of top active speakers to return
 * @returns {Object} 處理後的講者趨勢數據
 *                   Processed speaker trend data in the format:
 *                   {
 *                     years: number[],      // 排序後的年份列表 / Sorted list of years
 *                     speakers: string[],   // 所有講者名稱 / All speaker names
 *                     data: Array<{        // 前N名講者的詳細數據 / Detailed data for top N speakers
 *                       speaker: string,    // 講者名稱 / Speaker name
 *                       values: number[],   // 各年度演講次數 / Number of talks per year
*                        tags: Array<string>, // todo: 該講者的分類標籤
 *                       lectures: Array<{   // 該講者所有講題資訊 / All lecture information
 *                         title: string,    // 講題名稱 / Lecture title
 *                         year: number,     // 演講年份 / Year
 *                         community: String // 所屬社群 / Community,
 *                         tags: Array<string> // todo: 該講題的分類標籤
 *                       }>
 *                     }>
 *                   }
 */
export const processSpeakerTrends = async (yearlyData, topNSpeaker) => {
    // 1. 資料驗證 Data validation
    if (!yearlyData || typeof yearlyData !== 'object') {
        console.error('無效的年度資料 Invalid yearlyData:', yearlyData);
        return { years: [], speakers: [], data: [] };
    }        
    
    // 2. 使用全域轉換函數處理中文字 (簡體轉繁體)
    // Use global converter for Chinese text (Simplified to Traditional)
    const converter = {
        convert: text => window.convertToTraditional?.(text) || text
    };

    // 2.5 建立社群類型對應表
    const communityNameMap = new Map();  // 儲存社群名稱的標準化對應
    const communityIdMap = new Map();    // 儲存 ID 到標準化名稱的對應

    // 收集所有年度的 session_types
    Object.entries(yearlyData).forEach(([year, yearData]) => {
        if (year === 'speakers') return;  // 跳過非年度資料

        console.log(`處理 ${year} 年度的 session_types:`, yearData.session_types);

        if (Array.isArray(yearData.session_types)) {
            yearData.session_types.forEach(type => {
                if (type.id && type.zh?.name) {
                    const name = converter.convert(type.zh.name);
                    // 如果這個名稱還沒有標準化名稱，就用它自己作為標準名稱
                    if (!communityNameMap.has(name)) {
                        communityNameMap.set(name, name);
                    }
                    // 將 ID 對應到標準化名稱
                    communityIdMap.set(type.id, communityNameMap.get(name));
                }
            });
        }
    });

    // 印出社群對應關係
    console.log('各年度社群資訊：', Object.entries(yearlyData)
        .filter(([year, _]) => year !== 'speakers')
        .map(([year, data]) => ({
            年度: year,
            社群類型: data.session_types?.map(type => ({
                id: type.id,
                原始名稱: type.zh?.name,
                標準名稱: communityIdMap.get(type.id)
            }))
        }))
    );

    // 印出最終的對應表
    console.log('社群 ID 對照表：', {
        '標準化名稱列表': Array.from(communityNameMap.keys()),
        'ID對應標準名稱': Object.fromEntries(communityIdMap),
        '各社群的所有ID': Array.from(communityNameMap.keys()).map(standardName => ({
            社群: standardName,
            對應ID: Array.from(communityIdMap.entries())
                .filter(([_, name]) => name === standardName)
                .map(([id, _]) => id)
        }))
    });
    
    // 檢查年度資料中的社群 ID 使用情況
    Object.entries(yearlyData).forEach(([year, data]) => {
        if (year !== 'session_types' && year !== 'speakers' && data?.sessions) {
            const yearTypes = new Set(data.sessions.map(s => s.type));
            console.log(`${year} 年度使用的社群 ID:`, {
                年份: year,
                社群ID列表: Array.from(yearTypes),
                社群名稱對應: Array.from(yearTypes).map(id => ({
                    id,
                    name: communityIdMap.get(id) || '未知社群'
                }))
            });
        }
    });
    
    // 3. 建立講者名稱的映射，處理跨年度同名講者
    // Create speaker name mapping, handle cross-year speakers
    const uniqueSpeakers = new Map();  // 名稱 -> { name, years: Set<year>, ids: Set<id> }

    // 遍歷每一年的資料
    Object.entries(yearlyData).forEach(([year, yearData]) => {
        if (year === 'speakers' || !Array.isArray(yearData?.speakers)) return;
        
        yearData.speakers.forEach(speaker => {
            if (!speaker?.zh?.name) return;
            
            const rawName = speaker.zh.name;
            const convertedName = converter.convert(rawName);
            
            // 如果這個講者還沒有記錄，建立新的記錄
            if (!uniqueSpeakers.has(convertedName)) {
                uniqueSpeakers.set(convertedName, {
                    name: convertedName,
                    years: new Set(),
                    ids: new Set()
                });
            }
            
            // 更新講者資訊
            const speakerInfo = uniqueSpeakers.get(convertedName);
            speakerInfo.years.add(year);
            speakerInfo.ids.add(speaker.id);
        });
    });

    // 4. 建立ID到名稱的映射（用於後續處理）
    const speakerMap = new Map();  // ID -> 名稱
    uniqueSpeakers.forEach((info, name) => {
        info.ids.forEach(id => {
            speakerMap.set(id, name);
        });
    });

    console.log('講者統計:', {
        總講者數: uniqueSpeakers.size,
        跨年度講者: Array.from(uniqueSpeakers.values())
            .filter(info => info.years.size > 1)
            .map(info => ({
                name: info.name,
                years: Array.from(info.years).sort(),
                idCount: info.ids.size
            }))
    });
      // 計算原始總講者數和不重複講者數
    const rawSpeakerCount = Object.entries(yearlyData)
        .filter(([year, _]) => year !== 'speakers')
        .reduce((acc, [_, yearData]) => acc + (yearData?.speakers?.length || 0), 0);

    console.log(`原始講者總數 (含重複) Raw speaker count (with duplicates): ${rawSpeakerCount}`);
    console.log(`不重複講者數 Unique speakers: ${uniqueSpeakers.size}`);
    
    Object.entries(yearlyData).forEach(([year, yearData]) => {
        if (year !== 'speakers') {
            console.log(`處理年度 Processing year: ${year}, 該年度講者數 speakers this year: ${yearData?.speakers?.length || 0}`);
        }
    });

    // 6. 分析每位講者在各年度的活動次數和講題資訊
    // Analyze speaker activities count and lecture information for each year
    const speakerYearlyActivities = {};    // 講者年度活動統計
    const speakerLectures = {};            // 講者講題資訊
    const years = new Set();               // 收集所有年份
    
    Object.entries(yearlyData).forEach(([year, data]) => {
        if (year === 'speakers') return;

        const numericYear = parseInt(year);
        if (isNaN(numericYear)) return;

        years.add(numericYear);

        // 統計每個議程中講者的出現次數和收集講題資訊
        // Count speaker appearances and collect lecture information in each session
        if (!Array.isArray(data?.sessions)) return;
        data.sessions.forEach(session => {
            if (!Array.isArray(session?.speakers)) return;            // 從預先建立的對應表中獲取社群名稱
            const community = session.type ? communityIdMap.get(session.type) || '未分類' : '未分類';

            session.speakers.forEach(speakerId => {
                const speakerName = speakerMap.get(speakerId);
                if (!speakerName) return;

                // 初始化講者年度記錄
                // Initialize speaker yearly record
                if (!speakerYearlyActivities[speakerName]) {
                    speakerYearlyActivities[speakerName] = {};
                    speakerLectures[speakerName] = [];
                }

                // 累加該年度的演講次數
                // Increment talk count for the year
                speakerYearlyActivities[speakerName][numericYear] =
                    (speakerYearlyActivities[speakerName][numericYear] || 0) + 1;                // 收集講題資訊
                // Collect lecture information
                if (session.zh?.title) {
                    const lectureInfo = {
                        title: converter.convert(session.zh.title),
                        year: numericYear,
                        community: community
                    };
                    speakerLectures[speakerName].push(lectureInfo);

                    // 每50筆資料記錄一次，避免 log 太多
                    if (speakerLectures[speakerName].length % 50 === 1) {
                        console.log('講題資料範例:', {
                            講者: speakerName,
                            年份: numericYear,
                            講題資訊: lectureInfo,
                            社群Type: session.type,
                            對應社群: community,
                            目前講題數: speakerLectures[speakerName].length
                        });
                    }
                }
            });
        });
    });    
    
    // 7. 資料排序和最終處理
    // Sort and finalize data
    const sortedYears = Array.from(years).sort((a, b) => a - b); // 年份升序排列，由舊到新
    const sortedDescendingYears = Array.from(years).sort((a, b) => b - a); // 年份降序排列，由新到舊

    // 確保所有 uniqueSpeakers 中的講者都有對應的活動記錄
    uniqueSpeakers.forEach((info, speakerName) => {
        if (!speakerYearlyActivities[speakerName]) {
            speakerYearlyActivities[speakerName] = {};
            speakerLectures[speakerName] = [];
        }
    });

    // 建立講者總場次快取，避免重複計算
    const speakerTotalTalks = {};
    uniqueSpeakers.forEach((info, speaker) => {
        speakerTotalTalks[speaker] = sortedDescendingYears.reduce(
            (sum, year) => sum + (speakerYearlyActivities[speaker][year] || 0), 
            0
        );
    });

    // 將所有 uniqueSpeakers 轉為陣列進行排序
    const sortedSpeakers = Array.from(uniqueSpeakers.keys()).sort((speakerA, speakerB) => {
        const totalTalksA = speakerTotalTalks[speakerA];
        const totalTalksB = speakerTotalTalks[speakerB];
      
        // 總場次不同時，多的排前面
        if (totalTalksA !== totalTalksB) {
            return totalTalksB - totalTalksA;
        }

        // 總場次相同時，從新到舊比較每年的演講次數
        for (const year of sortedDescendingYears) {
            const yearTalksA = speakerYearlyActivities[speakerA][year] || 0;
            const yearTalksB = speakerYearlyActivities[speakerB][year] || 0;
            
            if (yearTalksA !== yearTalksB) {
                return yearTalksB - yearTalksA;
            }
        }
        
        // 所有年度都相同，則按講者名稱排序
        return speakerA.localeCompare(speakerB);
    });

    // 8. 轉換為最終資料格式
    // Transform into final data format
    const speakers = sortedSpeakers.map(speaker => ({
        speaker,
        values: sortedYears.map(year => speakerYearlyActivities[speaker][year] || 0),
        lectures: speakerLectures[speaker].sort((a, b) => a.year - b.year)  // 依年份排序講題
    }));    // 準備最終資料
    const finalData = speakers.slice(0, topNSpeaker);

    // 記錄最終資料統計
    console.log('資料處理完成：', {
        年份列表: sortedYears,
        講者總數: sortedSpeakers.length,
        資料筆數: finalData.length,
        社群統計: Array.from(new Set(finalData.flatMap(d => d.lectures.map(l => l.community))))
            .reduce((acc, comm) => {
                acc[comm] = finalData.reduce((sum, speaker) => 
                    sum + speaker.lectures.filter(l => l.community === comm).length, 0);
                return acc;
            }, {})
    });

    // 返回處理後的資料
    return {
        years: sortedYears,
        speakers: sortedSpeakers,
        data: finalData
    };
};

