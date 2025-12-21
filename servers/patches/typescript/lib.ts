/**
 * Patched lib.ts with I/O timing instrumentation
 * Based on: https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/lib.ts
 */

import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import os from 'os';
import { randomBytes } from 'crypto';
import { diffLines, createTwoFilesPatch } from 'diff';
import { minimatch } from 'minimatch';
import { normalizePath, expandHome } from './path-utils.js';
import { isPathWithinAllowedDirectories } from './path-validation.js';
import { measureIO } from './timing.js';

// Global allowed directories - set by the main module
let allowedDirectories: string[] = [];

// Function to set allowed directories from the main module
export function setAllowedDirectories(directories: string[]): void {
  allowedDirectories = [...directories];
}

// Function to get current allowed directories
export function getAllowedDirectories(): string[] {
  return [...allowedDirectories];
}

// Type definitions
interface FileInfo {
  size: number;
  created: Date;
  modified: Date;
  accessed: Date;
  isDirectory: boolean;
  isFile: boolean;
  permissions: string;
}

export interface SearchOptions {
  excludePatterns?: string[];
}

export interface SearchResult {
  path: string;
  isDirectory: boolean;
}

// Pure Utility Functions
export function formatSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  if (bytes === 0) return '0 B';

  const i = Math.floor(Math.log(bytes) / Math.log(1024));

  if (i < 0 || i === 0) return `${bytes} ${units[0]}`;

  const unitIndex = Math.min(i, units.length - 1);
  return `${(bytes / Math.pow(1024, unitIndex)).toFixed(2)} ${units[unitIndex]}`;
}

export function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n');
}

export function createUnifiedDiff(originalContent: string, newContent: string, filepath: string = 'file'): string {
  const normalizedOriginal = normalizeLineEndings(originalContent);
  const normalizedNew = normalizeLineEndings(newContent);

  return createTwoFilesPatch(
    filepath,
    filepath,
    normalizedOriginal,
    normalizedNew,
    'original',
    'modified'
  );
}

// Security & Validation Functions
export async function validatePath(requestedPath: string): Promise<string> {
  const expandedPath = expandHome(requestedPath);
  const absolute = path.isAbsolute(expandedPath)
    ? path.resolve(expandedPath)
    : path.resolve(process.cwd(), expandedPath);

  const normalizedRequested = normalizePath(absolute);

  const isAllowed = isPathWithinAllowedDirectories(normalizedRequested, allowedDirectories);
  if (!isAllowed) {
    throw new Error(`Access denied - path outside allowed directories: ${absolute} not in ${allowedDirectories.join(', ')}`);
  }

  try {
    // PATCHED: Wrap I/O with measureIO
    const realPath = await measureIO(() => fs.realpath(absolute));
    const normalizedReal = normalizePath(realPath);
    if (!isPathWithinAllowedDirectories(normalizedReal, allowedDirectories)) {
      throw new Error(`Access denied - symlink target outside allowed directories: ${realPath} not in ${allowedDirectories.join(', ')}`);
    }
    return realPath;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      const parentDir = path.dirname(absolute);
      try {
        // PATCHED: Wrap I/O with measureIO
        const realParentPath = await measureIO(() => fs.realpath(parentDir));
        const normalizedParent = normalizePath(realParentPath);
        if (!isPathWithinAllowedDirectories(normalizedParent, allowedDirectories)) {
          throw new Error(`Access denied - parent directory outside allowed directories: ${realParentPath} not in ${allowedDirectories.join(', ')}`);
        }
        return absolute;
      } catch {
        throw new Error(`Parent directory does not exist: ${parentDir}`);
      }
    }
    throw error;
  }
}

// File Operations - PATCHED with measureIO
export async function getFileStats(filePath: string): Promise<FileInfo> {
  // PATCHED: Wrap I/O with measureIO
  const stats = await measureIO(() => fs.stat(filePath));
  return {
    size: stats.size,
    created: stats.birthtime,
    modified: stats.mtime,
    accessed: stats.atime,
    isDirectory: stats.isDirectory(),
    isFile: stats.isFile(),
    permissions: stats.mode.toString(8).slice(-3),
  };
}

export async function readFileContent(filePath: string, encoding: string = 'utf-8'): Promise<string> {
  // PATCHED: Wrap I/O with measureIO
  return await measureIO(() => fs.readFile(filePath, encoding as BufferEncoding));
}

/**
 * Read a binary file and return its contents as a base64 string.
 * PATCHED: Wrap entire stream operation with measureIO for accurate I/O timing.
 */
export async function readFileAsBase64Stream(filePath: string): Promise<string> {
  // PATCHED: Wrap the entire stream read operation with measureIO
  return await measureIO(() => new Promise<string>((resolve, reject) => {
    const chunks: Buffer[] = [];
    const stream = createReadStream(filePath);

    stream.on('data', (chunk) => {
      chunks.push(chunk as Buffer);
    });

    stream.on('end', () => {
      const finalBuffer = Buffer.concat(chunks);
      resolve(finalBuffer.toString('base64'));
    });

    stream.on('error', (error) => {
      reject(error);
    });
  }));
}

export async function writeFileContent(filePath: string, content: string): Promise<void> {
  try {
    // PATCHED: Wrap I/O with measureIO
    await measureIO(() => fs.writeFile(filePath, content, { encoding: "utf-8", flag: 'wx' }));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'EEXIST') {
      const tempPath = `${filePath}.${randomBytes(16).toString('hex')}.tmp`;
      try {
        // PATCHED: Wrap I/O with measureIO
        await measureIO(() => fs.writeFile(tempPath, content, 'utf-8'));
        await measureIO(() => fs.rename(tempPath, filePath));
      } catch (renameError) {
        try {
          await fs.unlink(tempPath);
        } catch {}
        throw renameError;
      }
    } else {
      throw error;
    }
  }
}

// File Editing Functions
interface FileEdit {
  oldText: string;
  newText: string;
}

export async function applyFileEdits(
  filePath: string,
  edits: FileEdit[],
  dryRun: boolean = false
): Promise<string> {
  // PATCHED: Wrap I/O with measureIO
  const content = normalizeLineEndings(await measureIO(() => fs.readFile(filePath, 'utf-8')));

  let modifiedContent = content;
  for (const edit of edits) {
    const normalizedOld = normalizeLineEndings(edit.oldText);
    const normalizedNew = normalizeLineEndings(edit.newText);

    if (modifiedContent.includes(normalizedOld)) {
      modifiedContent = modifiedContent.replace(normalizedOld, normalizedNew);
      continue;
    }

    const oldLines = normalizedOld.split('\n');
    const contentLines = modifiedContent.split('\n');
    let matchFound = false;

    for (let i = 0; i <= contentLines.length - oldLines.length; i++) {
      const potentialMatch = contentLines.slice(i, i + oldLines.length);

      const isMatch = oldLines.every((oldLine, j) => {
        const contentLine = potentialMatch[j];
        return oldLine.trim() === contentLine.trim();
      });

      if (isMatch) {
        const originalIndent = contentLines[i].match(/^\s*/)?.[0] || '';
        const newLines = normalizedNew.split('\n').map((line, j) => {
          if (j === 0) return originalIndent + line.trimStart();
          const oldIndent = oldLines[j]?.match(/^\s*/)?.[0] || '';
          const newIndent = line.match(/^\s*/)?.[0] || '';
          if (oldIndent && newIndent) {
            const relativeIndent = newIndent.length - oldIndent.length;
            return originalIndent + ' '.repeat(Math.max(0, relativeIndent)) + line.trimStart();
          }
          return line;
        });

        contentLines.splice(i, oldLines.length, ...newLines);
        modifiedContent = contentLines.join('\n');
        matchFound = true;
        break;
      }
    }

    if (!matchFound) {
      throw new Error(`Could not find exact match for edit:\n${edit.oldText}`);
    }
  }

  const diff = createUnifiedDiff(content, modifiedContent, filePath);

  let numBackticks = 3;
  while (diff.includes('`'.repeat(numBackticks))) {
    numBackticks++;
  }
  const formattedDiff = `${'`'.repeat(numBackticks)}diff\n${diff}${'`'.repeat(numBackticks)}\n\n`;

  if (!dryRun) {
    const tempPath = `${filePath}.${randomBytes(16).toString('hex')}.tmp`;
    try {
      // PATCHED: Wrap I/O with measureIO
      await measureIO(() => fs.writeFile(tempPath, modifiedContent, 'utf-8'));
      await measureIO(() => fs.rename(tempPath, filePath));
    } catch (error) {
      try {
        await fs.unlink(tempPath);
      } catch {}
      throw error;
    }
  }

  return formattedDiff;
}

export async function tailFile(filePath: string, numLines: number): Promise<string> {
  const CHUNK_SIZE = 1024;
  // PATCHED: Wrap I/O with measureIO
  const stats = await measureIO(() => fs.stat(filePath));
  const fileSize = stats.size;

  if (fileSize === 0) return '';

  // PATCHED: Wrap I/O with measureIO
  const fileHandle = await measureIO(() => fs.open(filePath, 'r'));
  try {
    const lines: string[] = [];
    let position = fileSize;
    let chunk = Buffer.alloc(CHUNK_SIZE);
    let linesFound = 0;
    let remainingText = '';

    while (position > 0 && linesFound < numLines) {
      const size = Math.min(CHUNK_SIZE, position);
      position -= size;

      // PATCHED: Wrap I/O with measureIO
      const { bytesRead } = await measureIO(() => fileHandle.read(chunk, 0, size, position));
      if (!bytesRead) break;

      const readData = chunk.slice(0, bytesRead).toString('utf-8');
      const chunkText = readData + remainingText;

      const chunkLines = normalizeLineEndings(chunkText).split('\n');

      if (position > 0) {
        remainingText = chunkLines[0];
        chunkLines.shift();
      }

      for (let i = chunkLines.length - 1; i >= 0 && linesFound < numLines; i--) {
        lines.unshift(chunkLines[i]);
        linesFound++;
      }
    }

    return lines.join('\n');
  } finally {
    // PATCHED: Wrap I/O with measureIO
    await measureIO(() => fileHandle.close());
  }
}

export async function headFile(filePath: string, numLines: number): Promise<string> {
  // PATCHED: Wrap I/O with measureIO
  const fileHandle = await measureIO(() => fs.open(filePath, 'r'));
  try {
    const lines: string[] = [];
    let buffer = '';
    let bytesRead = 0;
    const chunk = Buffer.alloc(1024);

    while (lines.length < numLines) {
      // PATCHED: Wrap I/O with measureIO
      const result = await measureIO(() => fileHandle.read(chunk, 0, chunk.length, bytesRead));
      if (result.bytesRead === 0) break;
      bytesRead += result.bytesRead;
      buffer += chunk.slice(0, result.bytesRead).toString('utf-8');

      const newLineIndex = buffer.lastIndexOf('\n');
      if (newLineIndex !== -1) {
        const completeLines = buffer.slice(0, newLineIndex).split('\n');
        buffer = buffer.slice(newLineIndex + 1);
        for (const line of completeLines) {
          lines.push(line);
          if (lines.length >= numLines) break;
        }
      }
    }

    if (buffer.length > 0 && lines.length < numLines) {
      lines.push(buffer);
    }

    return lines.join('\n');
  } finally {
    // PATCHED: Wrap I/O with measureIO
    await measureIO(() => fileHandle.close());
  }
}

export async function searchFilesWithValidation(
  rootPath: string,
  pattern: string,
  allowedDirectories: string[],
  options: SearchOptions = {}
): Promise<string[]> {
  const { excludePatterns = [] } = options;
  const results: string[] = [];

  async function search(currentPath: string) {
    // PATCHED: Wrap I/O with measureIO
    const entries = await measureIO(() => fs.readdir(currentPath, { withFileTypes: true }));

    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);

      try {
        await validatePath(fullPath);

        const relativePath = path.relative(rootPath, fullPath);
        const shouldExclude = excludePatterns.some(excludePattern =>
          minimatch(relativePath, excludePattern, { dot: true })
        );

        if (shouldExclude) continue;

        if (minimatch(relativePath, pattern, { dot: true })) {
          results.push(fullPath);
        }

        if (entry.isDirectory()) {
          await search(fullPath);
        }
      } catch {
        continue;
      }
    }
  }

  await search(rootPath);
  return results;
}
